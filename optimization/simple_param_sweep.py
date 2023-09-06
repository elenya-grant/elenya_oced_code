import numpy as np
import os
import sys
import copy
import pandas as pd
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

def run_quick_lcoh(elec_eff,elec_cf,annual_h2,avg_stack_life_hrs,component_sizes,pf_param,pf_tool,return_details=False):
    component_sizes
    #assuming that optimizing in a non-policy case! so wind_frac = 1 doesn't matter
    sol,summary,price_breakdown,lcoh_breakdown = \
        pf_tool.run_lcoh_nostorage(pf_param,component_sizes,elec_eff,elec_cf,annual_h2,avg_stack_life_hrs,wind_frac = 1)
    # print('LCOH: {}'.format(sol['price']))
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
def test_battery(energy_from_renewables,best_Res,constraints,pf_param,pf_tool):
    from analysis.simple_dispatch import SimpleDispatch
    n_bat_sizes = 3
    bat_mults = [1,0.75,0.5,0.25,0]
    new_Res = best_Res.to_dict()
    new_res_cnt = {}
    S_elec = best_Res['electrolyzer_size_mw']
    dS_elec = constraints["unit_size_MW"]["electrolyzer"]
    dBat_MW = constraints['unit_size_MW']['battery']
    num_clusters = 1# S_elec/dS_elec

    bat = SimpleDispatch()
    bat.Nt = len(energy_from_renewables)
    electrical_power_signal = np.where(energy_from_renewables >S_elec*1000,S_elec*1000,energy_from_renewables)
    curtailed_power = np.where(energy_from_renewables >S_elec*1000,energy_from_renewables -S_elec*1000,0)
    sf_power = np.where(electrical_power_signal<(S_elec*0.1*1000),(S_elec*0.1*1000) - electrical_power_signal,0)

    bat.curtailment = curtailed_power
    bat.shortfall = sf_power
    battery_hour_storage_ub = constraints['ref_size_MWh']['battery']/constraints['max_size_MW']['battery']
    max_bat_size_kw = np.max([np.max(curtailed_power),np.max(sf_power),0.1*S_elec*1000])
    max_battery_storage_kWh = 4*max_bat_size_kw #TODO: fix hard-code
    for bi,bat_run in enumerate(bat_mults):
        if max_bat_size_kw > constraints['max_size_MW']['battery']*1000:
            max_bat_size_kw = constraints['max_size_MW']['battery']*1000
        if max_battery_storage_kWh > constraints['ref_size_MWh']['battery']*1000:
            max_battery_storage_kWh = constraints['ref_size_MWh']['battery']*1000
        max_bat_hrs = np.nan_to_num(max_battery_storage_kWh/max_bat_size_kw)
        if max_bat_hrs>battery_hour_storage_ub:
            max_bat_hrs = np.copy(battery_hour_storage_ub)
        bat = SimpleDispatch()
        bat.Nt = len(energy_from_renewables)
        bat.curtailment = curtailed_power
        bat.shortfall = sf_power    
        #TODO: add constraint for max hours
        bat.battery_storage = max_bat_hrs*max_bat_size_kw
        bat.charge_rate = max_bat_size_kw
        bat.discharge_rate = max_bat_size_kw
        battery_used, excess_energy, battery_SOC = bat.run()

        power_to_elec = electrical_power_signal+np.array(battery_used)
        # bat_hrs = np.nan_to_num(max_battery_storage_kWh/max_bat_size_kw)
        bat_size_mw = max_bat_size_kw/1000
        h2_results = run_elec(power_to_elec,S_elec,num_clusters)
        elec_eff = h2_results['Life: Average Efficiency [kWh/kg]']
        elec_cf = h2_results['Life: Capacity Factor [-]']
        
        annual_h2 = h2_results['Life: Average Annual Hydrogen Produced [kg]']
        avg_stack_life_hrs=h2_results['Life: Stack Life [hrs]']
        new_res_cnt['B: {}'.format(bi)]=new_Res.copy()
        new_res_cnt['B: {}'.format(bi)].update(h2_results)
        new_res_cnt['B: {}'.format(bi)].update({"battery_hrs": max_bat_hrs})
        new_res_cnt['B: {}'.format(bi)].update({"battery_size_mw":bat_size_mw})

        # new_Res.update({'battery_hrs':[]})
        # new_Res.update({'battery_size_mw':[]})
        cost_info = run_quick_lcoh(elec_eff,elec_cf,annual_h2,avg_stack_life_hrs,new_res_cnt['B: {}'.format(bi)],copy.copy(pf_param),copy.copy(pf_tool))
        new_res_cnt['B: {}'.format(bi)].update({'LCOH:' :cost_info[0]})
        max_bat_size_kw = bat_run*np.max(battery_used)
        max_battery_storage_kWh = bat_run*np.max(battery_SOC)
    return pd.DataFrame(new_res_cnt)
def learn_lcoh(elec_eff,elec_cf,annual_h2,avg_stack_life_hrs,component_sizes,pf_param,pf_tool,return_details=False):
    init_vals = {'Eff':elec_eff,'CF':elec_cf,'H2/year':annual_h2,'Stack Life':avg_stack_life_hrs}
    init_sizes = component_sizes.copy()
    res = run_quick_lcoh(elec_eff_test,elec_cf_test,annual_h2_test,avg_stack_life_hrs_test,component_sizes,copy.copy(pf_param),copy.copy(pf_tool),return_details=True)
    []
    
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
    #TODO: add in option to optimize electrolyzer size...
    
    S_wind_min = constraints["min_size_MW"]["wind"]
    S_wind_max = constraints["max_size_MW"]["wind"]
    dS_wind = constraints["unit_size_MW"]["wind"]
    S_wind_ref = constraints["ref_size_MW"]["wind"]
    S_wind_opt = np.arange(S_wind_min,S_wind_max + dS_wind,dS_wind)
    i_wRef = np.argwhere(S_wind_opt>S_wind_ref)[0][0]
    
    S_solar_min = constraints["min_size_MW"]["solar"]
    S_solar_max = constraints["max_size_MW"]["solar"]
    dS_solar = constraints["unit_size_MW"]["solar"]
    S_solar_ref = constraints["ref_size_MW"]["solar"]
    S_solar_opt = np.arange(S_solar_min,S_solar_max + dS_solar,dS_solar)
    i_sRef = np.argwhere(S_solar_opt>S_solar_ref)[0][0]

    S_elec_min = constraints["min_size_MW"]["electrolyzer"]
    S_elec_max = constraints["max_size_MW"]["electrolyzer"]
    dS_elec = constraints["unit_size_MW"]["electrolyzer"]
    S_elec_ref = constraints["ref_size_MW"]["electrolyzer"]
    S_elec_opt = np.arange(S_elec_min,S_elec_max+dS_elec,dS_elec)
    i_eRef = np.argwhere(S_elec_opt>S_elec_ref)[0][0]
    
    wind_percentiles_ub = [25,75]
    wind_percentiles_lb = [50,75]
    wind_sizes = ([dS_wind*(np.percentile(S_wind_opt[0:i_wRef],p)//dS_wind) for p in wind_percentiles_ub] + [S_wind_ref] +\
    [dS_wind*(np.percentile(S_wind_opt[i_wRef:],p)//dS_wind) for p in wind_percentiles_lb])
    solar_sizes = [S_solar_min] + ([dS_solar*(np.percentile(S_solar_opt[0:i_sRef],p)//dS_solar) for p in wind_percentiles_ub] +\
    [dS_solar*(np.percentile(S_solar_opt[i_sRef:],p)//dS_solar) for p in wind_percentiles_lb])
    
    most_solar_sizes = [S_solar_ref]*len(wind_sizes) + solar_sizes
    
    if optimize_electrolyzer:
        electrolzer_sizes = [S_elec_ref]*len(most_solar_sizes) + ([dS_elec*(np.percentile(S_elec_opt[0:i_eRef],p)//dS_elec) for p in wind_percentiles_ub] +\
        [dS_elec*(np.percentile(S_elec_opt[i_eRef:],p)//dS_elec) for p in wind_percentiles_lb])
    else:
        electrolzer_sizes = S_elec_ref*np.ones(len(most_solar_sizes))
    num_clusters = 1

    run_desc = 'W: '
    component_sizes={}
    component_sizes["battery_hrs"] = 0
    component_sizes["battery_size_mw"] = 0
    component_sizes["wind_size_mw"] = S_wind_ref
    component_sizes["solar_size_mw"] = S_solar_ref
    component_sizes["electrolyzer_size_mw"] = S_elec_ref
    S_elec = np.copy(S_elec_ref)
    component_sizes["desal_size_kg_pr_sec"] = S_elec_ref*18.311*pf_tool.kg_water_pr_kg_H2/3600
    h2_res = {}
    # lcoh_tracker = np.zeros(len(wind_sizes))
    # for wi,S_wind in enumerate(wind_sizes):
    for wi,S_elec in enumerate(electrolzer_sizes):
        # S_solar = most_solar_sizes[wi]
        if wi<len(most_solar_sizes):
            S_solar = most_solar_sizes[wi]
            num_clusters = 1
        elif wi==len(most_solar_sizes):
            i_best_solar = pd.DataFrame(h2_res).loc['LCOH:'].argmin()
            S_solar = pd.DataFrame(h2_res)[pd.DataFrame(h2_res).columns[i_best_solar]]['solar_size_mw']
        if wi<len(wind_sizes):
            S_wind = wind_sizes[wi]
            run_desc = 'W: '
        elif wi==len(wind_sizes):
            i_best_wind = pd.DataFrame(h2_res).loc['LCOH:'].argmin()
            S_wind = pd.DataFrame(h2_res)[pd.DataFrame(h2_res).columns[i_best_wind]]['wind_size_mw']
        else:
            run_desc = 'S: '
        
        if wi>len(most_solar_sizes):
            run_desc = 'E: '
            # num_clusters = S_elec/dS_elec
        
        
        component_sizes["wind_size_mw"] = S_wind
        component_sizes["solar_size_mw"] = S_solar
        k_wind = S_wind/S_wind_ref
        k_solar = S_solar/S_solar_ref
        wind_power_scaled = k_wind*wind_gen_kWh
        solar_power_scaled = k_solar*solar_gen_kWh
        energy_from_renewables = wind_power_scaled + solar_power_scaled
        electrical_power_signal = np.where(energy_from_renewables >S_elec*1000,S_elec*1000,energy_from_renewables)
        # curtailed_power = np.where(energy_from_renewables >S_elec*1000,energy_from_renewables -S_elec*1000,0)

        h2_results = run_elec(electrical_power_signal,S_elec,num_clusters)
        elec_eff = h2_results['Life: Average Efficiency [kWh/kg]']
        elec_cf = h2_results['Life: Capacity Factor [-]']
        
        annual_h2 = h2_results['Life: Average Annual Hydrogen Produced [kg]']
        avg_stack_life_hrs=h2_results['Life: Time Until Replacement [hrs]']#['Life: Stack Life [hrs]']
        cost_info = run_quick_lcoh(elec_eff,elec_cf,annual_h2,avg_stack_life_hrs,component_sizes,copy.copy(pf_param),copy.copy(pf_tool))

        # lcoh_tracker[wi] = cost_info[0]
        h2_res[run_desc + '{}'.format(wi)] = h2_results
        h2_res[run_desc + '{}'.format(wi)].update({'LCOH:' :cost_info[0]})
        h2_res[run_desc + '{}'.format(wi)].update(component_sizes)
        if wi == len(electrolzer_sizes):
            i_best_elec = pd.DataFrame(h2_res).loc['LCOH:'].argmin()
            S_elec = pd.DataFrame(h2_res)[pd.DataFrame(h2_res).columns[i_best_elec]]['electrolyzer_size_mw']
        
    []
    
    i_final_best = pd.DataFrame(h2_res).loc['LCOH:'].argmin()
    best_Res = pd.DataFrame(h2_res)[pd.DataFrame(h2_res).columns[i_final_best]]
    all_res = pd.DataFrame(h2_res)
    bat_Res = test_battery(energy_from_renewables,best_Res,constraints,copy.copy(pf_param),copy.copy(pf_tool))
    all_res = pd.concat([all_res,bat_Res],axis=1)
    i_best_overall = all_res.loc['LCOH:'].idxmin()
    final_best_Res = all_res[i_best_overall]
    # wind_res = pd.concat([pd.DataFrame(wind_h2_res),pd.DataFrame(lcoh_tracker,columns=['LCOH [$/kg]'],index=wind_sizes).T],axis=0)
    # if bat_Res.loc['LCOH:'].min() <best_Res.loc['LCOH:']:
    #     i_best_bat = bat_Res.loc['LCOH:'].idxmin()
    return final_best_Res,all_res
        
    
    []
    
    