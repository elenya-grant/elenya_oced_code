import numpy as np
import os
import sys
import copy
import pandas as pd
from matplotlib import pyplot as plt
# from params_new import *
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


# S_wind = 120*6
# wind_capex = wind_capex_pr_kW*S_wind*1000
# wind_opex = 0
# for d in denom:
#     wind_opex += (S_wind*1000*wind_opex_pr_kW)/d
# tot_cost = wind_opex + wind_capex

# alt_cost = wind_capex_pr_kW*1000 + sum(wind_opex_pr_kW*1000/d for d in denom)

# alt_cost*S_wind
# []
def estimate_lcoh(S_wind,S_elec):
    design_eff =54.60968574586311 #[kWh/kg]

    wind_capex = 1139
    wind_opex = 27
    solar_capex = 997
    solar_opex = 17
    dr = 0.0824
    plant_life = 30
    iF = 0.12
    indirect_cost = 0.42
    elec_uninstalled_capex = 340 #[$/kW]
    elec_fom = 12.8 #$/kW-year
    elec_vom = 1.3 #/MWh
    desal_capex = 32894 #$/kg/sec
    desal_opex = 4841 #$/kg/sec - year
    water_cost_USD_pr_gal = 0.004
    kg_h20_to_kg_h2 = 10
    elec_installed_capex = elec_uninstalled_capex*(1+iF)
    elec_indirect_capex = elec_installed_capex*indirect_cost
    elec_overnight_capex = elec_installed_capex + elec_indirect_capex
    years = np.arange(0,plant_life,1)

    unit_cost_wind_pr_MW = (1000*wind_capex) + sum((wind_opex*1000)/((1+dr)**y) for y in years)
    unit_cost_solar_pr_MW = (1000*solar_capex) + sum((solar_opex*1000)/((1+dr)**y) for y in years)
    unit_cost_elec_pr_MW = (1000*elec_overnight_capex) + sum((elec_fom*1000)/((1+dr)**y) for y in years)
    desal_size_pr_MW_elec = (1000/design_eff)*kg_h20_to_kg_h2/3600
    unit_cost_desal_pr_MWelec = (desal_capex*desal_size_pr_MW_elec) + sum((desal_opex*desal_size_pr_MW_elec)/((1+dr)**y) for y in years)
    full_unit_cost_elec_pr_MW = unit_cost_elec_pr_MW + unit_cost_desal_pr_MWelec

    annual_h2_rated = 8760*(S_elec*1000/design_eff)
    life_h2 = sum((annual_h2_rated*elec_cf)/((1+dr)**y) for y in years)

    dLCOH_dWind = unit_cost_wind_pr_MW/life_h2 #approx - change in $/kg divided by change in S_wind
    dLCOH_dSolar = unit_cost_solar_pr_MW/life_h2 #approx
    dLCOH_dElec = full_unit_cost_elec_pr_MW/(elec_cf*8760*(1000/design_eff))



def run_quick_lcoh(elec_eff,elec_cf,annual_h2,avg_stack_life_hrs,component_sizes,pf_param,pf_tool,return_details=False):
    component_sizes
    sol,summary,price_breakdown,lcoh_breakdown = \
        pf_tool.run_lcoh_nostorage(pf_param,component_sizes,elec_eff,elec_cf,annual_h2,avg_stack_life_hrs)
    print('LCOH: {}'.format(sol['price']))
    if return_details:
        return [sol["price"],lcoh_breakdown]
    else:
        return [sol["price"]]
def run_elec(electrical_power_signal,electrolyzer_size_mw,num_clusters):
    #electrical_power_signal - kW
    #electrolyzer_size_mw
    # from analysis.run_h2_PEM import run_h2_PEM
    from analysis.run_h2_PEM_mod import run_h2_PEM
    # H2_Results, h2_ts, h2_tot,energy_input_to_electrolyzer = run_h2_PEM(electrical_power_signal, electrolyzer_size_mw,
    #             useful_life, num_clusters,  electrolysis_scale, 
    #             pem_control_type,electrolyzer_direct_cost_kw_fake, user_defined_pem_param_dictionary,
    #             use_degradation_penalty, grid_connection_scenario,
    #             hydrogen_production_capacity_required_kgphr)
    H2_Results, h2_kg_pr_hr= run_h2_PEM(electrical_power_signal, electrolyzer_size_mw,
                useful_life, num_clusters,  electrolysis_scale, 
                pem_control_type,electrolyzer_direct_cost_kw_fake, user_defined_pem_param_dictionary,
                use_degradation_penalty, grid_connection_scenario,
                hydrogen_production_capacity_required_kgphr)
    return H2_Results
def learn_lcoh(elec_eff,elec_cf,annual_h2,avg_stack_life_hrs,component_sizes,pf_param,pf_tool,return_details=False):
    res_init = run_quick_lcoh(elec_eff,elec_cf,annual_h2,avg_stack_life_hrs,component_sizes,copy.copy(pf_param),copy.copy(pf_tool),return_details=True)
    init_vals = {'Eff':elec_eff,'CF':elec_cf,'H2/year':annual_h2,'Stack Life':avg_stack_life_hrs}
    init_sizes = component_sizes.copy()
    change_cf = np.arange(-0.1,elec_cf,0.1)
    change_eff = np.arange(-2,6,1)
    elec_cf_tests = np.arange(0.1,1.1,0.1)#elec_cf - change_cf
    elec_eff_tests = np.arange(30,60,2)#elec_eff-change_eff
    mult_annual_h2 = np.arange(0.5,1.6,0.1)
    annual_h2_tests = annual_h2*mult_annual_h2
    change_stack_life = 8760*np.arange(-2,3,1)
    avg_stack_life_hrs_tests = 8760*np.arange(1,10,1)#avg_stack_life_hrs - change_stack_life
    eff_cost = np.zeros(len(elec_eff_tests))
    cf_cost = np.zeros(len(elec_cf_tests))
    ah2_cost = np.zeros(len(mult_annual_h2))
    sr_cost = np.zeros(len(avg_stack_life_hrs_tests))
    sr_det = {}
    cf_det = {}
    eff_det = {}
    ah2_det = {}
    for i,elec_eff_test in enumerate(elec_eff_tests):
        res_eff = run_quick_lcoh(elec_eff_test,elec_cf,annual_h2,avg_stack_life_hrs,component_sizes,copy.copy(pf_param),copy.copy(pf_tool),return_details=True)
        eff_cost[i] = res_eff[0]
        eff_det[i] = res_eff[1]
    for i,elec_cf_test in enumerate(elec_cf_tests):
        res_cf = run_quick_lcoh(elec_eff,elec_cf_test,annual_h2,avg_stack_life_hrs,component_sizes,copy.copy(pf_param),copy.copy(pf_tool),return_details=True)
        cf_cost[i] = res_cf[0]
        cf_det[i] = res_cf[1]
    # for i,annual_h2_test in enumerate(annual_h2_tests):
    #     res_h2 = run_quick_lcoh(elec_eff,elec_cf,annual_h2_test,avg_stack_life_hrs,component_sizes,copy.copy(pf_param),copy.copy(pf_tool),return_details=True)
    #     ah2_cost[i] = res_h2[0]
    #     ah2_det[i] = res_h2[1]
    for i,avg_stack_life_hrs_test in enumerate(avg_stack_life_hrs_tests):
        res_sr = run_quick_lcoh(elec_eff,elec_cf,annual_h2,avg_stack_life_hrs_test,component_sizes,copy.copy(pf_param),copy.copy(pf_tool),return_details=True)
        sr_cost[i] = res_sr[0]
        sr_det[i] = res_sr[1]
    # dLCOH_dEff = (eff_cost - res_init[0])/(elec_eff_tests - elec_eff) #very minor impact
    # dLCOH_dcf = (cf_cost - res_init[0])/(elec_cf_tests - elec_cf) #HUGE IMPACT
    # dLCOH_dH2 = (ah2_cost - res_init[0])/(annual_h2_tests - annual_h2) #no impact?
    # dLCOH_dSr = (sr_cost - res_init[0])/(avg_stack_life_hrs_tests - avg_stack_life_hrs) #fairly minor impact
    dLCOH_dSr = np.diff(sr_cost)/np.diff(avg_stack_life_hrs_tests)
    # dLCOH_dH2 = np.diff(ah2_cost)/np.diff(annual_h2_tests)
    dLCOH_dcf = np.diff(cf_cost)/np.diff(elec_cf_tests)
    dLCOH_dEff = np.diff(eff_cost)/np.diff(elec_eff_tests)

    sizes_d = {'battery_size_mw':0,'battery_hrs':0}
    sizes_test = {'battery_size_mw':0,'battery_hrs':0,'wind_size_mw':int(component_sizes['wind_size_mw']),'solar_size_mw':int(component_sizes['solar_size_mw']),'electrolyzer_size_mw':int(component_sizes['electrolyzer_size_mw']),'desal_size_kg_pr_sec':component_sizes['desal_size_kg_pr_sec']}
    # sizes_keys = ['battery_size_mw','battery_hrs','wind_size_mw','solar_size_mw','electrolyzer_size_mw','desal_size_kg_pr_sec']
    sizes_keys = ['wind_size_mw','solar_size_mw','electrolyzer_size_mw']
    # sizes_vals = np.array([0,0,int(component_sizes['wind_size_mw']),int(component_sizes['solar_size_mw']),int(component_sizes['electrolyzer_size_mw']),component_sizes['desal_size_kg_pr_sec']])
    sizes_vals = np.array([int(component_sizes['wind_size_mw']),int(component_sizes['solar_size_mw']),int(component_sizes['electrolyzer_size_mw'])])
    #component_sizes["desal_size_kg_pr_sec"] = S_elec*18.311*pf_tool.kg_water_pr_kg_H2/3600
    scale_up = 0.5*np.identity(len(sizes_vals))
    scale_down = 0.5*np.identity(len(sizes_vals))
    inc_sizes = sizes_vals*np.ones((3,1)) + sizes_vals*np.ones((3,1))*scale_up
    dec_sizes = sizes_vals*np.ones((3,1)) - sizes_vals*np.ones((3,1))*scale_up
    inc_desal = inc_sizes[:,2]*(18.311*pf_tool.kg_water_pr_kg_H2/3600)
    dec_desal = dec_sizes[:,2]*(18.311*pf_tool.kg_water_pr_kg_H2/3600)
    cost_psize = np.zeros(len(sizes_vals))
    cost_dsize = np.zeros(len(sizes_vals))
    psize_det = {}
    dsize_det = {}
    for i in range(len(inc_sizes)):
        ind=dict(zip(sizes_keys,inc_sizes[i]))
        sizes_d.update(ind)
        sizes_d.update({'desal_size_kg_pr_sec':inc_desal[i]})
        res_inc_size = run_quick_lcoh(elec_eff,elec_cf,annual_h2,avg_stack_life_hrs,sizes_d,copy.copy(pf_param),copy.copy(pf_tool),return_details=True)
        cost_psize[i] = res_inc_size[0]
        psize_det[i]=res_inc_size[1]
    for i in range(len(dec_sizes)):
        ind=dict(zip(sizes_keys,dec_sizes[i]))
        sizes_d.update(ind)
        sizes_d.update({'desal_size_kg_pr_sec':dec_desal[i]})
        res_dec_size = run_quick_lcoh(elec_eff,elec_cf,annual_h2,avg_stack_life_hrs,sizes_d,copy.copy(pf_param),copy.copy(pf_tool),return_details=True)
        cost_dsize[i] = res_dec_size[0]
        dsize_det[i]=res_dec_size[1]
    dSize_inc = np.diag(inc_sizes) - sizes_vals
    dSize_dec = np.diag(dec_sizes) - sizes_vals
    dSize_both = np.diag(inc_sizes) - np.diag(dec_sizes)
    dLCOH_both = cost_psize - cost_dsize
    dLCOH_inc = cost_psize-res_init[0]
    dLCOH_dec = cost_dsize-res_init[0]

    grad_inc = dLCOH_inc/dSize_inc
    grad_dec = dLCOH_dec/dSize_dec
    grad_both = dLCOH_both/dSize_both
    fig,ax = plt.subplots(3,2)
    fig.set_figheight(9)
    fig.set_figwidth(6)
    ax[0,0].scatter(elec_cf_tests,cf_cost,color='blue')
    ax[0,0].scatter(elec_cf,res_init[0],marker = '*',color='red')
    ax[0,0].set_xlabel('CF')

    ax[1,0].scatter(elec_eff_tests,eff_cost,color='blue')
    ax[1,0].scatter(elec_eff,res_init[0],marker = '*',color='red')
    ax[1,0].set_xlabel('Efficiency [kWh/kg]')

    ax[2,0].scatter(avg_stack_life_hrs_tests/8760,sr_cost,color='blue')
    ax[2,0].scatter(avg_stack_life_hrs/8760,res_init[0],marker = '*',color='red')
    ax[2,0].set_xlabel('Stack Life [yrs]')

    ax[0,1].scatter([dec_sizes[0,0],inc_sizes[0,0]],[cost_dsize[0],cost_psize[0]],color='blue')
    ax[0,1].scatter(sizes_vals[0],res_init[0],marker = '*',color='red')
    ax[0,1].set_xlabel('Wind Size [MW]')

    ax[1,1].scatter([dec_sizes[1,1],inc_sizes[1,1]],[cost_dsize[1],cost_psize[1]],color='blue')
    ax[1,1].scatter(sizes_vals[1],res_init[0],marker = '*',color='red')
    ax[1,1].set_xlabel('Solar Size [MW]')

    ax[2,1].scatter([dec_sizes[2,2],inc_sizes[2,2]],[cost_dsize[2],cost_psize[2]],color='blue')
    ax[2,1].scatter(sizes_vals[2],res_init[0],marker = '*',color='red')
    ax[2,1].set_xlabel('Elec Size [MW]')

    fig.tight_layout()
    []

def param_sweep(
    wind_gen_kWh,
    solar_gen_kWh,
    constraints,
    pf_param,
    pf_tool,
    n_sizes_pr_component = 5,
    # kg_h2o_pr_kg_h2,
    # wind_losses,
    # solar_losses,
    # stack_size_MW,
    tr=0.1
):

    num_components = 3
    n_runs_max = n_sizes_pr_component*num_components
    S_wind_min = constraints["min_size_MW"]["wind"]
    S_wind_max = constraints["max_size_MW"]["wind"]
    dS_wind = constraints["unit_size_MW"]["wind"]
    S_wind_ref = constraints["ref_size_MW"]["wind"]
    
    S_solar_min = constraints["min_size_MW"]["solar"]
    S_solar_max = constraints["max_size_MW"]["solar"]
    dS_solar = constraints["unit_size_MW"]["solar"]
    S_solar_ref = constraints["ref_size_MW"]["solar"]
    

    S_elec_min = constraints["min_size_MW"]["electrolyzer"]
    S_elec_max = constraints["max_size_MW"]["electrolyzer"]
    dS_elec = constraints["unit_size_MW"]["electrolyzer"]
    S_elec_ref = constraints["ref_size_MW"]["electrolyzer"]
    num_clusters = 1

    component_sizes={}
    component_sizes["battery_hrs"] = 0
    component_sizes["battery_size_mw"] = 0
    # component_sizes["wind_size_mw"] = S_wind_ref
    # component_sizes["solar_size_mw"] = S_solar_ref
    # component_sizes["electrolyzer_size_mw"] = S_elec_ref
    # component_sizes["desal_size_kg_pr_sec"] = S_elec_ref*18.311*pf_tool.kg_water_pr_kg_H2/3600

    
    S_wind = np.copy(S_wind_ref)
    S_solar = np.copy(S_solar_ref)
    S_elec = np.copy(S_elec_ref)

    # S_wind_track = np.zeros(n_runs_max+1)
    # S_solar_track = np.zeros(n_runs_max+1)
    # S_elec_track = np.zeros(n_runs_max+1)
    # lcoh_track = np.zeros(n_runs_max+1)
    grad_track = np.zeros(n_runs_max+1)

    dbg_h2_res = {}
    
    init_size_change = np.array([30*dS_wind,200*dS_solar, 2*dS_elec])
    test_sizes = init_size_change*np.identity(3)
    test_sizes = np.concatenate([init_size_change*np.identity(3),(-1*init_size_change)*np.identity(3)])
    
    init_sizes = [S_wind_ref,S_solar_ref,S_elec_ref]
    new_sizes = init_sizes
    #start with elec
    #move to wind
    #then do solar
    # for i in range(n_sizes_pr_component+1):
    S_wind_track = np.zeros(len(test_sizes)+1)
    S_solar_track = np.zeros(len(test_sizes)+1)
    S_elec_track = np.zeros(len(test_sizes)+1)
    lcoh_track = np.zeros(len(test_sizes)+1)
    for i in range(len(test_sizes)+1):
        S_wind_track[i]=S_wind
        S_solar_track[i] = S_solar
        S_elec_track[i] = S_elec
        energy_from_renewables = wind_gen_kWh+solar_gen_kWh
        electrical_power_signal = np.where(energy_from_renewables >S_elec*1000,S_elec*1000,energy_from_renewables)
        component_sizes["wind_size_mw"] = S_wind
        component_sizes["solar_size_mw"] = S_solar
        component_sizes["electrolyzer_size_mw"] = S_elec
        component_sizes["desal_size_kg_pr_sec"] = S_elec*18.311*pf_tool.kg_water_pr_kg_H2/3600
        h2_results = run_elec(electrical_power_signal,S_elec,num_clusters)
        dbg_h2_res[i] = h2_results
        elec_eff = h2_results['Life: Average Efficiency [kWh/kg]']
        elec_cf = h2_results['Life: Capacity Factor [-]']
        
        annual_h2 = h2_results['Life: Average Annual Hydrogen Produced [kg]']
        avg_stack_life_hrs=h2_results['Life: Stack Life [hrs]']
        #run LCOH
        #TODO: remove below
        learn_lcoh(elec_eff,elec_cf,annual_h2,avg_stack_life_hrs,component_sizes,copy.copy(pf_param),copy.copy(pf_tool))
        cost_info = run_quick_lcoh(elec_eff,elec_cf,annual_h2,avg_stack_life_hrs,component_sizes,copy.copy(pf_param),copy.copy(pf_tool))
        lcoh=cost_info[0]
        lcoh_track[i] = cost_info[0]
        if i==len(test_sizes):
            new_sizes = init_sizes
        else:
            new_sizes = init_sizes + test_sizes[i]
        S_wind = new_sizes[0]
        S_solar = new_sizes[1]
        S_elec = new_sizes[2]
    init_df = pd.DataFrame(dict(zip(['wind','solar','electrolyzer','LCOH'],[S_wind_track,S_solar_track,S_elec_track,lcoh_track])))

    dSolar = init_df['solar']-init_df['solar'][0]
    dWind = init_df['wind']-init_df['wind'][0]
    dElec = init_df['electrolyzer']-init_df['electrolyzer'][0]
    dLCOH = init_df['LCOH']-init_df['LCOH'][0]
    i_dSolar = np.argwhere(dSolar.values!=0)[:,0]
    i_dWind = np.argwhere(dWind.values!=0)[:,0]
    i_dElec = np.argwhere(dElec.values!=0)[:,0]
    dC_dS_Solar = dLCOH.values[i_dSolar]/dSolar.values[i_dSolar]
    dC_dS_Wind = dLCOH.values[i_dWind]/dWind.values[i_dWind]
    dC_dS_elect = dLCOH.values[i_dElec]/dElec.values[i_dElec]

    for i in range(n_runs_max+1):
        S_wind_track[i]=S_wind
        S_solar_track[i] = S_solar
        S_elec_track[i] = S_elec
        energy_from_renewables = wind_gen_kWh+solar_gen_kWh
        electrical_power_signal = np.where(energy_from_renewables >S_elec*1000,S_elec*1000,energy_from_renewables)
        component_sizes["wind_size_mw"] = S_wind
        component_sizes["solar_size_mw"] = S_solar
        component_sizes["electrolyzer_size_mw"] = S_elec
        component_sizes["desal_size_kg_pr_sec"] = S_elec*18.311*pf_tool.kg_water_pr_kg_H2/3600
        h2_results = run_elec(electrical_power_signal,S_elec,num_clusters)
        dbg_h2_res[i] = h2_results
        elec_eff = h2_results['Life: Average Efficiency [kWh/kg]']
        elec_cf = h2_results['Life: Capacity Factor [-]']
        
        annual_h2 = h2_results['Life: Average Annual Hydrogen Produced [kg]']
        avg_stack_life_hrs=h2_results['Life: Stack Life [hrs]']
        #run LCOH
        cost_info = run_quick_lcoh(elec_eff,elec_cf,annual_h2,avg_stack_life_hrs,component_sizes,copy.copy(pf_param),copy.copy(pf_tool))
        lcoh=cost_info[0]
        lcoh_track[i] = cost_info[0]
        
        if i==0:
            best_lcoh = cost_info[0]
            S_elec +=dS_elec
        else:
            dLCOH = lcoh_track[i] - lcoh_track[i-1]
            dSize = S_elec - S_elec_track[i-1]
            grad = dLCOH/dSize
            delta_size = grad*dS_elec*2000#np.sign(grad)*np.sign(dSize)*dS_elec
            S_elec = np.ceil(S_elec_track[i] - delta_size)
            
            best_lcoh = np.min([lcoh,best_lcoh])
            if S_elec<S_elec_min:
                S_elec = S_elec_min
            if S_elec > S_elec_max:
                S_elec = S_elec_max
        []
    []
#         else:
#             dLCOH = lcoh_track[i] - lcoh_track[i-1]
#             dSize = S_elec - S_elec_track[i-1]
#             # grad = dLCOH/dSize
#             grad = dLCOH/dSize
#             grad_track[i]=grad
#             if i==1:
#                 alpha = np.abs((dSize*grad))/(grad*grad)
#             else:
#                 alpha = np.abs((dSize*(grad-grad_track[i-1])))/((grad-grad_track[i-1])**2)
#             delta_size = grad*alpha
#             # grad = dSize/dLCOH
#             # delta_size = grad*dS_elec*1000
#             S_elec = S_elec_track[i] - delta_size
#             if S_elec<S_elec_min:
#                 S_elec = S_elec_min
#             if S_elec > S_elec_max:
#                 S_elec = S_elec_max
#             print(S_elec)
                
#         best_lcoh = np.min([lcoh,best_lcoh])
        
#     []
#     for i in range(n_runs_max+1):
#         S_wind_track[i]=S_wind
#         S_solar_track[i] = S_solar
#         S_elec_track[i] = S_elec

#         wind_power_scaled = (S_wind/S_wind_ref)*wind_gen_kWh
#         solar_power_scaled = (S_solar/S_solar_ref)*solar_gen_kWh
        
#         energy_from_renewables = wind_power_scaled+solar_power_scaled
        
#         electrical_power_signal = np.where(energy_from_renewables >S_elec*1000,S_elec*1000,energy_from_renewables)
        
#         # run electrolyzer
#         h2_results = run_elec(electrical_power_signal,S_elec,num_clusters)
#         dbg_h2_res[i] = h2_results
#         #update component sizes
#         component_sizes["wind_size_mw"] = S_wind
#         component_sizes["solar_size_mw"] = S_solar
#         component_sizes["electrolyzer_size_mw"] = S_elec
#         component_sizes["desal_size_kg_pr_sec"] = S_elec*18.311*pf_tool.kg_water_pr_kg_H2/3600

#         elec_eff = h2_results['Life: Average Efficiency [kWh/kg]']
#         elec_cf = h2_results['Life: Capacity Factor [-]']
        
#         annual_h2 = h2_results['Life: Average Annual Hydrogen Produced [kg]']
#         avg_stack_life_hrs=h2_results['Life: Stack Life [hrs]']
#         #run LCOH
#         cost_info = run_quick_lcoh(elec_eff,elec_cf,annual_h2,avg_stack_life_hrs,component_sizes,copy.copy(pf_param),copy.copy(pf_tool))
#         lcoh=cost_info[0]
#         lcoh_track[i] = cost_info[0]
#         best_lcoh = np.min([lcoh,best_lcoh])
        
        
#     component_sizes["battery_size_mw"] = S_bat
#     component_sizes["battery_hrs"] = S_bat_capac/S_bat

# #
# # S_wind=[]
# # S_solar=[]
# # S_bat=[]
# # S_bat_capac = []
# # S_elec = []

# # elec_capex_pr_MW = 1000*(BOP_CapEx_pr_kW + (electrolyzer_overnight_CapEx_pr_kW*(1+indirect_electrolyzer_costs_percent)))
# # elec_opex_pr_MW = 1000*(BOP_OpEx_pr_kW + electrolyzer_FOM_pr_kW)
# # C_wind = wind_capex_pr_kW*1000 + sum(wind_opex_pr_kW*1000/d for d in denom) #[$/MW]
# # C_solar = pv_capex_pr_kW*1000 + sum(pv_opex_pr_kW*1000/d for d in denom) #[$/MW]
# # C_bat_cr = bat_capex_pr_kW*1000 + sum(battery_opex_perc*bat_capex_pr_kW*1000/d for d in denom)
# # C_bat_capac = bat_capex_pr_kWh*1000 + sum(battery_opex_perc*bat_capex_pr_kWh*1000/d for d in denom)
# # C_elec = elec_capex_pr_MW + sum(elec_opex_pr_MW/d for d in denom)

# # c = [C_wind,C_solar,C_bat_cr,C_bat_capac,C_elec]
# # x = [S_wind,S_solar,S_bat,S_bat_capac,S_elec]


