# from analysis.hopp_for_h2 import hopp_for_h2
# from analysis.simple_dispatch import SimpleDispatch
# # from analysis.hopp_for_h2_floris import hopp_for_h2_floris
# from run_h2_PEM import run_h2_PEM
import numpy as np
# import pandas as pd
#import yaml
def hydrogen_storage_capacity_cost_calcs(H2_Results,electrolyzer_size_mw,storage_type):

    hydrogen_average_output_kgprhr = np.mean(H2_Results['hydrogen_hourly_production'])
    hydrogen_surplus_deficit = H2_Results['hydrogen_hourly_production'] - hydrogen_average_output_kgprhr

    hydrogen_storage_soc = []
    for j in range(len(hydrogen_surplus_deficit)):
        if j == 0:
            hydrogen_storage_soc.append(hydrogen_surplus_deficit[j])
        else:
            hydrogen_storage_soc.append(hydrogen_storage_soc[j-1]+hydrogen_surplus_deficit[j])
            
    hydrogen_storage_capacity_kg = np.max(hydrogen_storage_soc) - np.min(hydrogen_storage_soc)
    h2_LHV = 119.96
    h2_HHV = 141.88
    hydrogen_storage_capacity_MWh_LHV = hydrogen_storage_capacity_kg*h2_LHV/3600
    hydrogen_storage_capacity_MWh_HHV = hydrogen_storage_capacity_kg*h2_HHV/3600
             
    # hydrogen_storage_duration_hr = hydrogen_storage_capacity_MWh_LHV/electrolyzer_size_mw/H2_Results['electrolyzer_total_efficiency']
    hydrogen_storage_duration_hr = hydrogen_storage_capacity_MWh_LHV/electrolyzer_size_mw/H2_Results['electrolyzer_avg_efficiency']
    
    equation_year_CEPCI = 603.1
    model_year_CEPCI = 607.5
    
    if storage_type == 'Salt cavern' or storage_type == 'salt cavern' or storage_type == 'salt' or storage_type == 'Salt':
        if hydrogen_storage_capacity_MWh_HHV <= 120293:
            base_capacity_MWh_HHV = 120293
            base_cost_USDprkg = 17.04
            scaling_factor = 0.611
            storage_cost_USDprkg = model_year_CEPCI/equation_year_CEPCI*base_capacity_MWh_HHV*base_cost_USDprkg*(hydrogen_storage_capacity_MWh_HHV/base_capacity_MWh_HHV)**scaling_factor/hydrogen_storage_capacity_MWh_HHV
            # status_message = 'Hydrogen storage model complete.\nStorage capacity: ' + str(hydrogen_storage_capacity_kg/1000) + ' metric tonnes. \nStorage cost: ' + str(storage_cost_USDprkg) + ' $/kg'
        else:
            storage_cost_USDprkg = model_year_CEPCI/equation_year_CEPCI*17.04
            # status_message = 'Hydrogen storage model complete.\nStorage capacity: ' + str(hydrogen_storage_capacity_kg/1000) + ' metric tonnes. \nStorage cost: ' + str(storage_cost_USDprkg) + ' $/kg'
    elif storage_type == 'Lined rock cavern' or storage_type == 'lined rock cavern' or storage_type == 'Lined rock' or storage_type == 'lined rock':
        if hydrogen_storage_capacity_MWh_HHV <= 119251:
            base_capacity_MWh_HHV = 119251
            base_cost_USDprkg = 42.42
            scaling_factor = 0.7016
            storage_cost_USDprkg = model_year_CEPCI/equation_year_CEPCI*base_capacity_MWh_HHV*base_cost_USDprkg*(hydrogen_storage_capacity_MWh_HHV/base_capacity_MWh_HHV)**scaling_factor/hydrogen_storage_capacity_MWh_HHV
            # status_message = 'Hydrogen storage model complete'
        else:
            storage_cost_USDprkg = model_year_CEPCI/equation_year_CEPCI*42.42
            # status_message = 'Hydrogen storage model complete.\nStorage capacity: ' + str(hydrogen_storage_capacity_kg/1000) + ' metric tonnes. \nStorage cost: ' + str(storage_cost_USDprkg) + ' $/kg'
    elif storage_type == 'Buried pipes' or storage_type == 'buried pipes' or storage_type == 'pipes' or storage_type == 'Pipes':
        if hydrogen_storage_capacity_MWh_HHV <= 4085:
            base_capacity_MWh_HHV = 4085
            base_cost_USDprkg = 521.34
            scaling_factor = 0.9592
            storage_cost_USDprkg = model_year_CEPCI/equation_year_CEPCI*base_capacity_MWh_HHV*base_cost_USDprkg*(hydrogen_storage_capacity_MWh_HHV/base_capacity_MWh_HHV)**scaling_factor/hydrogen_storage_capacity_MWh_HHV
            # status_message = 'Hydrogen storage model complete'
        else:
            storage_cost_USDprkg = model_year_CEPCI/equation_year_CEPCI*521.34
            # status_message = 'Hydrogen storage model complete.\nStorage capacity: ' + str(hydrogen_storage_capacity_kg/1000) + ' metric tonnes. \nStorage cost: ' + str(storage_cost_USDprkg) + ' $/kg'
    else:
        if hydrogen_storage_capacity_MWh_HHV <= 4085:
            base_capacity_MWh_HHV = 4085
            base_cost_USDprkg = 521.34
            scaling_factor = 0.9592
            storage_cost_USDprkg = model_year_CEPCI/equation_year_CEPCI*base_capacity_MWh_HHV*base_cost_USDprkg*(hydrogen_storage_capacity_MWh_HHV/base_capacity_MWh_HHV)**scaling_factor/hydrogen_storage_capacity_MWh_HHV
            # status_message = 'Hydrogen storage model complete'
        else:
            storage_cost_USDprkg = model_year_CEPCI/equation_year_CEPCI*521.34
            # status_message = 'Error: Please enter a valid hydrogen storage type. Otherwise, assuming buried pipe (location agnostic) hydrogen storage.\nStorage capacity: ' \
            #     + str(hydrogen_storage_capacity_kg/1000) + ' metric tonnes. \nStorage cost: ' + str(storage_cost_USDprkg) + ' $/kg'
    if hydrogen_storage_capacity_MWh_HHV==0:
        storage_cost_USDprkg=0
    return(hydrogen_storage_capacity_kg,hydrogen_storage_duration_hr,storage_cost_USDprkg)
    #hydrogen_storage_cost_USDprkg
    # return(hydrogen_average_output_kgprhr,hydrogen_storage_capacity_kg,hydrogen_storage_capacity_MWh_HHV,hydrogen_storage_duration_hr,storage_cost_USDprkg,status_message)
    
def calc_desal_costs(electrolyzer_size_mw,kWh_per_kgH2=54.6):
    #kWh_per_kgH2=55.5?
        print("calculating desasl costs...")
        #Equations from desal_model
        #Values for CAPEX and OPEX given as $/(kg/s)
        #Source: https://www.nrel.gov/docs/fy16osti/66073.pdf
        #Assumed density of recovered water = 997 kg/m^3

        m3_water_per_kg_h2 = 0.01
        # #desired fresh water flow rate [m^3/hr]
        desal_sys_size = electrolyzer_size_mw * (1000/kWh_per_kgH2) * m3_water_per_kg_h2 #m^3
        # desal_sys_size = electrolyzer_size_mw * (10/kWh_per_kgH2)
        desal_opex = 4841 * (997 * desal_sys_size / 3600) # Output in USD/yr
        # desal_opex = 1340.6880555555556*desal_sys_size
        desal_capex = 32894 * (997 * desal_sys_size / 3600) # Output in USD
        # desal_capex =9109.810555555556*desal_sys_size

        return desal_capex,desal_opex

def custom_hydrogen_storage_cost(self,hydrogen_storage_capacity_kg,hydrogen_storage_capex_pr_kg):
    []
def custom_hydrogen_compression_costs(self):
    pass
def custom_water_consumption(self):
    #50.2L/kg H2 for cooling
    #35.1L/kg-LH2 for liquidfaction
    #15.4L/kg-Hs for electrolyzer


    pass
def custom_bop_power_consumption(self):
    pass
def custom_electrolyzer_costs(self):
    pass
