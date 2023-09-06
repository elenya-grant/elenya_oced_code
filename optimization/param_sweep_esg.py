import numpy as np
import os
import sys
import copy
import pandas as pd
from matplotlib import pyplot as plt
import scipy
import time

def new_params_for_func(X,a,b,c,d):
    cf = (a*(X**2)) + (b*X) + (c*(X**0.5)) + d
    return cf
def two_params_for_func(X,a,b,c,d,e,f):
    x,y = X
    cf = (a*(x**2)) + (b*(y**2)) + x*y*c + (d*(x**0.5)) + (e*(y**0.5)) + f
    return cf

def func(X,a,b,c,d,e):
    x,y=X
    return [(2*a*x) + (c*y) + (0.5*d*(x**(-1/2))),
    (2*b*y) + (c*x) + (0.5*e*(y**(-1/2)))]


parent_dir = os.path.abspath('')
sys.path.append('..')
# from analysis.run_PEM_master import run_PEM_clusters
# from tools.elec_cost_scaling_tools import *
user_defined_pem_param_dictionary = {
    "Modify BOL Eff": False,
    "BOL Eff [kWh/kg-H2]": [],
    "Modify EOL Degradation Value": True,
    "EOL Rated Efficiency Drop": 10,
}
electrolyzer_direct_cost_kw_fake=100
useful_life = 30
use_degradation_penalty = True
electrolysis_scale = 'Centralized'
pem_control_type = 'basic'
grid_connection_scenario = 'off-grid'
hydrogen_production_capacity_required_kgphr=[]
design_eff = 54.60968574586311

    



def run_quick_lcoh(elec_eff,elec_cf,annual_h2,avg_stack_life_hrs,component_sizes,pf_param,pf_tool,return_details=False):
    sol,summary,price_breakdown,lcoh_breakdown = \
        pf_tool.run_lcoh_nostorage(pf_param,component_sizes,elec_eff,elec_cf,annual_h2,avg_stack_life_hrs,wind_frac=1)
    # print('LCOH: {}'.format(sol['price']))
    if return_details:
        return [sol["price"],lcoh_breakdown]
    else:
        return [sol["price"]]
def run_elec(electrical_power_signal,electrolyzer_size_mw,num_clusters):
    
    from analysis.run_h2_PEM_mod import run_h2_PEM
    
    H2_Results, h2_kg_pr_hr= run_h2_PEM(electrical_power_signal, electrolyzer_size_mw,
                useful_life, num_clusters,  electrolysis_scale, 
                pem_control_type,electrolyzer_direct_cost_kw_fake, user_defined_pem_param_dictionary,
                use_degradation_penalty, grid_connection_scenario,
                hydrogen_production_capacity_required_kgphr)
    return H2_Results
def coupled_wind_solar(S_elec_ref,S_solar_ref,S_wind_ref,wind_gen_kWh,solar_gen_kWh,lcoh_tools,return_details = False):
    # start = time.perf_counter()
    
    pf_param = copy.copy(lcoh_tools[0])
    pf_tool = copy.copy(lcoh_tools[1])
    sizes_d = {'battery_size_mw':0,'battery_hrs':0,'electrolyzer_size_mw':S_elec_ref} #new
    desal_size_kg = S_elec_ref*(1000/design_eff)*pf_tool.kg_water_pr_kg_H2/3600
    sizes_d.update({"desal_size_kg_pr_sec":desal_size_kg})

    ky_range =np.array([1/4,1/2,1,3/2,9/4,3])
    kx_range =np.array([1/4,1/2,1,3/2,9/4,3])
    x0 = S_solar_ref
    y0 = S_wind_ref
    z0 = S_elec_ref
    i=0
    x_vals = np.zeros(len(kx_range)**2)
    y_vals = np.zeros(len(kx_range)**2)
    lcoh_vals = np.zeros(len(kx_range)**2) #new
    
    if return_details:
        res_tracker = pd.DataFrame()
        
    
    best_lcoh = 100000

    for kx in kx_range:
        x=kx*z0
        power_x = (x/x0)*solar_gen_kWh
        sizes_d.update({'solar_size_mw':x}) #new
        for ky in ky_range:
            y=ky*z0
            sizes_d.update({'wind_size_mw':y}) #new
            power_y = (y/y0)*wind_gen_kWh
            res = run_elec(power_x + power_y,z0,1)
            x_vals[i] = x
            y_vals[i] = y
            
            lcoh = run_quick_lcoh(res['Life: Average Efficiency [kWh/kg]'],res['Life: Capacity Factor [-]'],res['Life: Average Annual Hydrogen Produced [kg]'],res['Life: Time Until Replacement [hrs]'],sizes_d,copy.copy(pf_param),copy.copy(pf_tool))
            lcoh_vals[i] = lcoh[0]
            res.update({'solar_size_mw':x,'wind_size_mw':y,'lcoh':lcoh[0]})

            if lcoh[0]<=best_lcoh:
                    best_lcoh = lcoh[0]
            if return_details:
                
                res_tracker = pd.concat([res_tracker,pd.Series(res,name=i)],axis=1)
            else:
                if lcoh[0]<=best_lcoh:
                    best_lcoh = lcoh[0]
                    res_tracker = res.copy()

            i+=1

    c_kxky_coeff,c_kxky_cov = scipy.optimize.curve_fit(two_params_for_func,(x_vals/z0,y_vals/z0),lcoh_vals,p0=(1.0,1.0,1.0,1.0,1.0,1.0))
    a,b,c,d,e,f = c_kxky_coeff
    if return_details:
        i_min = res_tracker.loc['lcoh'].idxmin()
        root_est =[res_tracker[i_min]['solar_size_mw']/S_elec_ref,res_tracker[i_min]['wind_size_mw']/S_elec_ref]
    else:
        root_est = [res_tracker['solar_size_mw']/S_elec_ref,res_tracker['wind_size_mw']/S_elec_ref]
        
    # root = scipy.optimize.fsolve(func,[1,1],args=(a,b,c,d,e))
    root = scipy.optimize.fsolve(func,root_est,args=(a,b,c,d,e))
    # x_sol = root[0]
    # y_sol = root[1]
    # min_lcoh = two_params_for_func((x_sol,y_sol),*c_kxky_coeff)
    min_lcoh = two_params_for_func((root[0],root[1]),*c_kxky_coeff)
    if min_lcoh<=best_lcoh:
        success_flag = True
        #success
    else:
        success_flag = False
        #fail

    # S_opt_solar = root[0]*z0
    # S_opt_wind = root[1]*z0

    # end= time.perf_counter()
    # print("Took {} sec to get curve".format((np.round(end-start,3))))
    # return root,[S_opt_solar,S_opt_wind],res_tracker,c_kxky_coeff,success_flag
    return root,res_tracker,c_kxky_coeff,success_flag

def electrolyzer_size_opt(z0,k_xz,x0,solar_gen_kWh,k_yz,y0,wind_gen_kWh,lcoh_tools,kz_max,return_details=False):
    pf_param = copy.copy(lcoh_tools[0])
    pf_tool = copy.copy(lcoh_tools[1])

    y=k_yz*z0
    x=k_xz*z0
    power_y = (y/y0)*wind_gen_kWh
    power_x = (x/x0)*solar_gen_kWh
    res = run_elec(power_x + power_y,z0,1)
    rated_annual_h2_pr_MWz = 8760*(1000/design_eff)
    cf = res['Life: Capacity Factor [-]']
    stack_life = res['Life: Time Until Replacement [hrs]']
    avg_eff = res['Life: Average Efficiency [kWh/kg]']

    # kz_range = np.array([0.2,1,3]) #([0.2 , 0.5 , 1.  , 1.5 , 2.25, 3.  ])
    # kz_range = np.arange(0.25,3.25,0.5)
    if kz_max>1.25:
        kz_range = np.concatenate([np.array([1,0.25,0.75]),np.arange(1.25,kz_max+0.25,0.25)]) #
    else:
        kz_range = np.concatenate([np.array([1,0.25,0.75,kz_max]),np.arange(0.1,kz_max,0.25)])
    # kz_range = np.arange(0.25,1,3)

    z_vals = kz_range*z0

    sizes_d = {'battery_size_mw':0,'battery_hrs':0} #new
    size_keys = ['electrolyzer_size_mw','solar_size_mw','wind_size_mw','desal_size_kg_pr_sec']
    lcoh_vals = np.zeros(len(kz_range))
    res_tracker = pd.DataFrame()
    for i,z in enumerate(z_vals):    
        
        y=k_yz*z
        x=k_xz*z
        # power_y = (y/y0)*wind_gen_kWh
        # power_x = (x/x0)*solar_gen_kWh
        size_vals = [z,x,y,z*(1000/design_eff)*pf_tool.kg_water_pr_kg_H2/3600]
        sizes_d.update(dict(zip(size_keys,size_vals)))
        # res = run_elec(power_x + power_y,z,1)
        annual_h2_est = z*rated_annual_h2_pr_MWz*cf
        lcoh = run_quick_lcoh(avg_eff,cf,annual_h2_est ,stack_life,sizes_d,copy.copy(pf_param),copy.copy(pf_tool))
        lcoh_vals[i] = lcoh[0]
        if i>0:
            if np.abs(lcoh_vals[i]-lcoh_vals[i-1]) < 0.001:
                # print("break")
                break

        # res.update({'solar_size_mw':x,'wind_size_mw':y,'electrolyzer_size_mw':z,'lcoh':lcoh[0]})
        # res_tracker= pd.concat([res_tracker,pd.Series(res,name=i)],axis=1)
    []
    
    i_min = np.argmin(lcoh_vals[np.nonzero(lcoh_vals)])
    best_zsize = z_vals[i_min]
    return best_zsize
    
    # c_kz_coeff,c_kz_cov = scipy.optimize.curve_fit(new_params_for_func,kz_range,lcoh_vals,p0=(1.0,1.0,1.0,1.0))
    
    # if sum(np.sign(np.diff(lcoh_vals)))==0:
    #     min_in_range = True
    # n_runs = 10
    # new_cnt = np.arange(i,i+n_runs,1)
    # z_vals = np.concatenate([z_vals,np.zeros(len(new_cnt)+1)])
    # lcoh_vals = np.concatenate([lcoh_vals,np.zeros(len(new_cnt)+1)])
    # for i in new_cnt:
        # delta_c0 = lcoh_vals[i-1]-lcoh_vals[i-2]
        # delta_c1 = lcoh_vals[i]-lcoh_vals[i-1]
        # delta_z0 = z_vals[i-1]-z_vals[i-2]
        # delta_z1 = z_vals[i]-z_vals[i-1]

        # y=k_yz*z_vals[i]
        # x=k_xz*z_vals[i]
        # power_y = (y/y0)*wind_gen_kWh
        # power_x = (x/x0)*solar_gen_kWh
        # size_vals = [z_vals[i],x,y,z_vals[i]*(1000/design_eff)*pf_tool.kg_water_pr_kg_H2/3600]
        # sizes_d.update(dict(zip(size_keys,size_vals)))
        # res = run_elec(power_x + power_y,z_vals[i],1)
        # lcoh = run_quick_lcoh(res['Life: Average Efficiency [kWh/kg]'],res['Life: Capacity Factor [-]'],res['Life: Average Annual Hydrogen Produced [kg]'],res['Life: Time Until Replacement [hrs]'],sizes_d,copy.copy(pf_param),copy.copy(pf_tool))
        # lcoh_vals[i] = lcoh[0]
        # res.update({'solar_size_mw':x,'wind_size_mw':y,'electrolyzer_size_mw':z_vals[i],'lcoh':lcoh[0]})

    #     #start gradient
    #     grad_0 = (lcoh_vals[i-1]-lcoh_vals[i-2])/(z_vals[i-1]-z_vals[i-2])
    #     grad_1 = (lcoh_vals[i]-lcoh_vals[i-1])/(z_vals[i]-z_vals[i-1])
    #     # grad_minus = delta_c0/delta_z0
    #     # grad_now = delta_c1/delta_z1
    #     delta_grad = grad_1-grad_0
    #     delta_z = z_vals[i]-z_vals[i-1]

    #     dz = np.abs(delta_z*delta_grad)/(delta_grad*delta_grad)
    #     z_next = z_vals[i]-dz*grad_1
    #     if z_next<0:
    #         z_next=100
    #     z_vals[i+1] = z_next

    #     # dz = np.abs(delta_z1*delta_grad)/(delta_grad*delta_grad)
    #     # z_next = z_vals[2]-dz*grad_now
    # []
    # # desal_size_kg = z*(1000/design_eff)*pf_tool.kg_water_pr_kg_H2/3600
    
    # # sizes_d.update({"desal_size_kg_pr_sec":desal_size_kg})
    

def param_sweep(
    wind_gen_kWh,
    solar_gen_kWh,
    constraints,
    pf_param,
    pf_tool,
    optimize_electrolyzer,
    save_sweep_results,
    # kg_h2o_pr_kg_h2,
    # wind_losses,
    # solar_losses,
    # stack_size_MW,
    tr=0.1
):
    #print('im here')
    
    x=constraints["ref_size_MW"]["solar"]
    y=constraints["ref_size_MW"]["wind"]
    z=constraints["ref_size_MW"]["electrolyzer"]
    
    lcoh_tools = [copy.copy(pf_param),copy.copy(pf_tool)]
    # [solar_elec_ratio,wind_elec_ratio],[S_opt_solar,S_opt_wind],res_tracker,c_kxky_coeff,success_flag = \
    #     coupled_wind_solar(S_elec_ref,S_solar_ref,S_wind_ref,wind_gen_kWh,solar_gen_kWh,lcoh_tools,return_details=False)
    # [k_xz,k_yz],[x_opt,y_opt],res_tracker,c_kxky_coeff,success_flag = \
    #     coupled_wind_solar(z,x,y,wind_gen_kWh,solar_gen_kWh,lcoh_tools,return_details=False)
    # start= time.perf_counter()
    [k_xz,k_yz],res_tracker,c_kxky_coeff,success_flag = \
        coupled_wind_solar(z,x,y,wind_gen_kWh,solar_gen_kWh,lcoh_tools,return_details=save_sweep_results)
    

    best_res = {'battery_size_mw':0,'battery_hrs':0}#new
    
    if not success_flag:
        # best_res = res_tracker.copy()
        if save_sweep_results:
            i_min = res_tracker.loc['lcoh'].idxmin()
            k_xz = res_tracker.loc['solar_size_mw'][i_min]/z
            k_yz = res_tracker.loc['wind_size_mw'][i_min]/z
        else:
            k_xz = res_tracker['solar_size_mw']/z
            k_yz = res_tracker['wind_size_mw']/z
        # min_lcoh = res_tracker['lcoh']
    min_lcoh = two_params_for_func((k_xz,k_yz),*c_kxky_coeff)
    # else:
    #     # min_lcoh = two_params_for_func((k_xz,k_yz),*c_kxky_coeff)
    #     x_opt = k_xz*z
    #     y_opt = k_yz*z
    z_max = np.min([constraints["max_size_MW"]["wind"]/k_yz,constraints["max_size_MW"]["solar"]/k_xz])
    kz_max = z_max/z
    if optimize_electrolyzer:
        z_opt = electrolyzer_size_opt(z,k_xz,x,solar_gen_kWh,k_yz,y,wind_gen_kWh,lcoh_tools,kz_max,return_details=False)
    else:
        z_opt = z
    # best_res['estimated_lcoh'] = min_lcoh
    
    
    constraints["max_size_MW"]["wind"]/k_yz
    num_wind_units = np.floor(k_yz*z_opt/constraints["unit_size_MW"]["wind"])
    num_solar_units = np.floor(k_xz*z_opt/constraints["unit_size_MW"]["solar"])

    wind_size_mw = num_wind_units*constraints["unit_size_MW"]["wind"]
    solar_size_mw = num_solar_units*constraints["unit_size_MW"]["solar"]


    # num_electrolyzer_units = np.floor(z_opt/constraints["unit_size_MW"]["electrolyzer"])
    best_res.update({'electrolyzer_size_mw':z_opt})
    
    # end= time.perf_counter()
    # print("Took {} sec to run optimization".format((np.round(end-start,3))))
    
    best_res['wind_size_mw'] = wind_size_mw
    best_res['solar_size_mw'] = solar_size_mw
    best_res['success'] = success_flag
    best_res['opt_est_lcoh']=min_lcoh
    desal_size_kg = z_opt*(1000/design_eff)*pf_tool.kg_water_pr_kg_H2/3600
    best_res.update({"desal_size_kg_pr_sec":desal_size_kg})
        # [k_xz,k_yz],[x_opt,y_opt],res_tracker,c_kxky_coeff,success_flag = \
        # coupled_wind_solar(z,x,y,wind_gen_kWh,solar_gen_kWh,lcoh_tools,return_details=True)
        # res_tracker['lcoh'].idxmin()
        # res_tracker
    all_res={'res_track':res_tracker,'best_res':best_res,'opt_res':{'c_kxky_coeff':c_kxky_coeff,'opt_wind_to_elec':k_yz,'opt_solar_to_elec':k_xz}}
    return pd.Series(best_res),pd.Series(all_res)