# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 12:09:20 2022

@author: ereznic2
"""
import pandas as pd
import numpy as np
import os.path




def hydrogen_LCA_singlescenario_ProFAST(cambiumdata_filepath,grid_connection_scenario,hydrogen_annual_output_kgperyr,energy_from_grid_kWh,system_life=30):


    kg_to_MT_conv = 0.001 # Converion from kg to metric tonnes

    cambium_data = pd.read_csv(cambiumdata_filepath,index_col = None,header = 5,usecols = ['lrmer_co2_c','lrmer_ch4_c','lrmer_n2o_c','lrmer_co2_p','lrmer_ch4_p','lrmer_n2o_p','lrmer_co2e_c','lrmer_co2e_p','lrmer_co2e'])
    
    cambium_data = cambium_data.reset_index().rename(columns = {'index':'Interval','lrmer_co2_c':'LRMER CO2 combustion (kg-CO2/MWh)','lrmer_ch4_c':'LRMER CH4 combustion (g-CH4/MWh)','lrmer_n2o_c':'LRMER N2O combustion (g-N2O/MWh)',\
                                                  'lrmer_co2_p':'LRMER CO2 production (kg-CO2/MWh)','lrmer_ch4_p':'LRMER CH4 production (g-CH4/MWh)','lrmer_n2o_p':'LRMER N2O production (g-N2O/MWh)','lrmer_co2e_c':'LRMER CO2 equiv. combustion (kg-CO2e/MWh)',\
                                                  'lrmer_co2e_p':'LRMER CO2 equiv. production (kg-CO2e/MWh)','lrmer_co2e':'LRMER CO2 equiv. total (kg-CO2e/MWh)'})
    
    cambium_data['Interval']=cambium_data['Interval']+1
    cambium_data = cambium_data.set_index('Interval')   
    
    # Calculate hourly grid emissions factors of interest. If we want to use different GWPs, we can do that here. The Grid Import is an hourly data i.e., in MWh
    cambium_data['Total grid emissions (kg-CO2e)'] = energy_from_grid_kWh * cambium_data['LRMER CO2 equiv. total (kg-CO2e/MWh)'] / 1000
    cambium_data['Scope 2 (combustion) grid emissions (kg-CO2e)'] = energy_from_grid_kWh  * cambium_data['LRMER CO2 equiv. combustion (kg-CO2e/MWh)'] / 1000
    cambium_data['Scope 3 (production) grid emissions (kg-CO2e)'] = energy_from_grid_kWh  * cambium_data['LRMER CO2 equiv. production (kg-CO2e/MWh)'] / 1000
    
    # Sum total emissions
    scope2_grid_emissions_sum = cambium_data['Scope 2 (combustion) grid emissions (kg-CO2e)'].sum()*system_life*kg_to_MT_conv
    scope3_grid_emissions_sum = cambium_data['Scope 3 (production) grid emissions (kg-CO2e)'].sum()*system_life*kg_to_MT_conv
    
    h2prod_sum=hydrogen_annual_output_kgperyr*system_life*kg_to_MT_conv


    if grid_connection_scenario == 'hybrid-grid' :
        # Calculate grid-connected electrolysis emissions/ future cases should reflect targeted electrolyzer electricity usage
        electrolysis_Scope3_EI =  scope3_grid_emissions_sum/h2prod_sum # + (wind_capex_EI + solar_pv_capex_EI + battery_EI) * (scope3_ren_sum/h2prod_sum) * g_to_kg_conv + ely_stack_capex_EI # kg CO2e/kg H2
        electrolysis_Scope2_EI =  scope2_grid_emissions_sum/h2prod_sum 
        electrolysis_Scope1_EI = 0
        electrolysis_total_EI  = electrolysis_Scope1_EI + electrolysis_Scope2_EI + electrolysis_Scope3_EI 
        electrolysis_total_EI_policy_grid = electrolysis_total_EI # - (wind_capex_EI + solar_pv_capex_EI + battery_EI) * (scope3_ren_sum/h2prod_sum)  * g_to_kg_conv 
        electrolysis_total_EI_policy_offgrid = 0 #(wind_capex_EI + solar_pv_capex_EI + battery_EI) * (scope3_ren_sum/h2prod_sum)  * g_to_kg_conv + ely_stack_capex_EI
    elif grid_connection_scenario == 'grid-only':
        # Calculate grid-connected electrolysis emissions
        electrolysis_Scope3_EI = scope3_grid_emissions_sum/h2prod_sum # + ely_stack_capex_EI # kg CO2e/kg H2
        electrolysis_Scope2_EI = scope2_grid_emissions_sum/h2prod_sum 
        electrolysis_Scope1_EI = 0
        electrolysis_total_EI = electrolysis_Scope1_EI + electrolysis_Scope2_EI + electrolysis_Scope3_EI
        electrolysis_total_EI_policy_grid = electrolysis_total_EI 
        electrolysis_total_EI_policy_offgrid = 0
    elif grid_connection_scenario == 'off-grid':    
        # Calculate renewable only electrolysis emissions        
        electrolysis_Scope3_EI = 0#(wind_capex_EI + solar_pv_capex_EI + battery_EI) * (scope3_ren_sum/h2prod_sum)  * g_to_kg_conv + ely_stack_capex_EI # kg CO2e/kg H2
        electrolysis_Scope2_EI = 0
        electrolysis_Scope1_EI = 0
        electrolysis_total_EI = electrolysis_Scope1_EI + electrolysis_Scope2_EI + electrolysis_Scope3_EI
        electrolysis_total_EI_policy_offgrid = electrolysis_total_EI 
        electrolysis_total_EI_policy_grid = 0
    
    return(electrolysis_total_EI_policy_grid,electrolysis_total_EI_policy_offgrid)    


