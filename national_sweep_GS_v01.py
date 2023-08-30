import os
import sys
#sys.path.append('')
from dotenv import load_dotenv
import pandas as pd
from hybrid.sites.site_info  import SiteInfo
import copy
import numpy as np

import warnings
from analysis.hybrid_plant_tools import run_hybrid_plant
# from scipy import interpolate
# from analysis.optimize_run_profast_for_hydrogen import opt_run_profast_for_hydrogen
from analysis.GS_v01_PF_for_H2 import LCOH_Calc
from analysis.GS_v01_set_PF_params import init_profast
# from analysis.GS_v01_PF_for_H2_Storage import LCOH_Calc
from analysis.PEM_H2_LT_electrolyzer_Clusters import PEM_H2_Clusters
# from analysis.opt_run_profast_for_h2_transmission import run_profast_for_h2_transmission
# from analysis.additional_cost_tools import hydrogen_storage_capacity_cost_calcs
# from analysis.additional_cost_tools import hydrogen_storage_capacity_auto_calc_NEW,hydrogen_storage_cost_calc_NEW
# from analysis.electrolyzer_tools import electrolyzer_tools as pem_tool
# from tools.floris_2_csv import csv_from_floris
# from tools.hydrogen_storage_tools_GS import calc_hydrogen_storage_CapEx,calc_hydrogen_storage_size_kg
#from math import cos, asin, sqrt, pi
warnings.filterwarnings("ignore")
# import yaml

class opt_national_sweep:
#Step 1: initialize inputs
#     a: costs
#     b: design
#https://www.eia.gov/electricity/state/
#     c: load profile (opt)
    def __init__(self,model_params):
        #50k points is 2km resolution
        #can use 10-50km resolution
        pem_cost_cases = ['Mod 19','Mod 18']
        
        # self.parent_path = os.path.abspath('') + '/'
        self.main_dir = os.path.abspath('') + '/'
        # self.input_info_dir = self.main_dir + 'input_params/'
        self.resource_directory = self.main_dir  + 'resource_files/'
        
        
        self.input_info_dir = self.main_dir + 'input_info/'
        self.electrolyzer_installed_cost_opt=[]
        self.model_params = model_params#pd.read_csv(default_input_file,index_col = 'Variable')

        # DEFAULTS #
        self.resource_year = model_params['plant_general']['resource_year']
        # self.annual_hydrogen_required_kg = 66000*1000 #metric ton-> kg
        self.end_of_life_eff_drop = model_params['electrolyzer']['config']['EOL_eff_drop']#for electrolyzer
        
        self.plant_life = model_params['plant_general']['plant_life']

        self.floris=model_params['turbine']['config']['floris']
        model_params['turbine']['config']['turb_rating_mw']
        model_params['turbine']['constraints']
        self.run_wind = True
        self.run_solar = True
        self.save_sweep_results = model_params['outputs']['save_sweep_results']
        self.output_dir = model_params['outputs']['save_output_dir']
        #66,000 metric tons of hydrogen per year.
        # self.electrolyzer_capacity_MW = 720 #800
        # target_electrolyzer_cf = 0.6
        # hourly_kgh2_target = self.annual_hydrogen_required_kg/(target_electrolyzer_cf*8760)
        # required_electrolyzer_capacity_MW = hourly_kgh2_target/18.11

        # self.annual_hydrogen_required_kg = self.target_electrolyzer_cf*8760*(self.electrolyzer_capacity_MW*18.11)
        self.stack_size_MW = model_params['electrolyzer']['config']['stack_size_MW']
        self.pem_control_type = model_params['electrolyzer']['config']["pem_control_type"]
        self.h2_reqd_kg_per_hr = model_params["plant_general"]["hydrogen_production_capacity_required_kgphr"]
        
        
        turb_rating_mw = model_params['turbine']['config']['turb_rating_mw']
        turb_name = model_params['turbine']['config']['name']
        self.wind_scenario = self.init_wind(turb_rating_mw,turb_name)
        self.kg_water_pr_kg_H2 = model_params["electrolyzer"]["config"]["kg_H20_pr_kg_H2"]
        self.elec_rated_eff_kWh_pr_kg = self.init_electrolyzer_efficiency() #initializes: self.elec_rated_eff_kWh_pr_kg
        # self.wind_scenario = self.init_wind(turb_rating_mw) #TODO: double check if using bespoke ref turbine
        model_params["electrolyzer"]["config"]['kWh_pr_kg_design'] = self.elec_rated_eff_kWh_pr_kg
        self.turb_rating_mw = turb_rating_mw
        self.use_degradation_penalty = model_params["electrolyzer"]["config"]["use_degradation_penalty"]
        self.pem_input_dictionary = self.init_electrolyzer_inputs(model_params)
        self.grid_connection_scenario = model_params["plant_general"]["grid_connection_scenario"]
        self.electrolysis_scale = model_params["plant_general"]["electrolysis_scale"]
    

    def run_site_outline(self,site_info,model_params):

        policy_cases = model_params["simulation"]["policy_cases"]
        cost_scenarios = model_params["simulation"]["cost_scenarios"]
        storage_scenarios = model_params["simulation"]["storage_types"]
        
        pf_params=init_profast(model_params)
        cost_year = model_params["simulation"]['re_cost_year_scenario']
        policy_desc_opt = 'no_policy'
        re_cost_desc_opt = 'Moderate'
        model_params['optimization']['optimization_cost_cases']
        model_params['optimization']['optimziation_policy_cases']
        
        pf_tool_opt= LCOH_Calc(model_params,cost_year,policy_desc_opt,re_cost_desc_opt)
        

        site_obj = self.make_site(site_info['latitude'],site_info['longitude'],self.wind_scenario['Tower Height'])
        
        # opt_res = self.run_optimizer() #will output optimal wind, solar, battery, and electrolyzer size
        optimal_sizes = self.run_optimizer(site_obj,model_params,pf_params,pf_tool_opt)
        solar_gen_kWh,wind_gen_kWh,hybrid_plant = self.run_hopp(site_obj,optimal_sizes["wind_size_mw"],optimal_sizes["solar_size_mw"])
        energy_from_renewables = solar_gen_kWh + wind_gen_kWh
        wind_frac = np.sum(wind_gen_kWh)/np.sum(energy_from_renewables)
        optimal_sizes["electrolyzer_size_mw"]
        if optimal_sizes["battery_size_mw"]>0:
            electrical_power_input_kWh = self.run_battery(optimal_sizes["electrolyzer_size_mw"],optimal_sizes["battery_size_mw"],optimal_sizes["battery_hrs"],energy_from_renewables)
        else:
            electrical_power_input_kWh = np.where(energy_from_renewables >optimal_sizes["electrolyzer_size_mw"]*1000,optimal_sizes["electrolyzer_size_mw"]*1000,energy_from_renewables)
            # battery_used = np.zeros(len(energy_from_renewables))
        desal_size_kg_pr_sec = self.calc_desal_size(optimal_sizes["electrolyzer_size_mw"])
        optimal_sizes.update({"desal_size_kg_pr_sec":desal_size_kg_pr_sec})
        H2_res,hydrogen_hourly_production = self.run_electrolyezr_briefly(optimal_sizes["electrolyzer_size_mw"],electrical_power_input_kWh)
        hydrogen_storage_size_kg = self.calc_hydrogen_storage_size(hydrogen_hourly_production)
        # desal_size_kg_pr_sec = self.calc_desal_size(electrolyzer_size_mw)
        optimal_sizes.update({"hydrogen_storage_size_kg":hydrogen_storage_size_kg})
        elec_eff_kWh_pr_kg = H2_res['Life: Average Efficiency [kWh/kg]']
        elec_cf = H2_res['Life: Capacity Factor [-]']
        annual_hydrogen_kg = H2_res['Life: Average Annual Hydrogen Produced [kg]']
        # avg_stack_life_hrs=H2_res['Life: Stack Life [hrs]']
        avg_stack_life_hrs = H2_res['Life: Time Until Replacement [hrs]']

        lcoh_breakdown_tracker = pd.DataFrame()
        price_breakdown_tracker = {}
        lcoh_h2_tracker = []
        lcoh_full_tracker = []

        storage_desc = []
        renewable_cost_scenario = []
        policy_scenario = []
        for storage_type in storage_scenarios:
            hydrogen_storage_capex_pr_kg = self.calc_hydrogen_storage_CapEx(hydrogen_storage_size_kg,storage_type,model_params)
            hydrogen_storage_opex_pr_kg=model_params["hydrogen_storage_cases"][storage_type]["opex_per_kg"]
            for re_cost_desc in cost_scenarios:
                for policy_desc in policy_cases:
                    # print('{}-{}-{}'.format(storage_type,re_cost_desc,policy_desc))

                    storage_desc.append(storage_type)
                    renewable_cost_scenario.append(re_cost_desc)
                    policy_scenario.append(policy_desc)

                    pf_params=init_profast(model_params)
                    pf_tool= LCOH_Calc(model_params,cost_year,policy_desc,re_cost_desc)
                    #run lcoh breakdown without hydrogen storage
                    sol_h2,summary_h2,price_breakdown_h2,lcoh_breakdown_h2 = \
                    pf_tool.run_lcoh_nostorage(copy.copy(pf_params),optimal_sizes,elec_eff_kWh_pr_kg,elec_cf,annual_hydrogen_kg,avg_stack_life_hrs,wind_frac)
                    # pf_tool.run_lcoh_nostorage(copy.copy(pf_params),optimal_sizes,elec_eff_kWh_pr_kg,elec_cf,annual_hydrogen_kg,avg_stack_life_hrs)
                    price_breakdown_tracker['{}-{}-{}'.format(re_cost_desc,policy_desc,'no_storage')] = price_breakdown_h2
                    lcoh_h2_tracker.append(sol_h2['price'])
                    lcoh_breakdown_tracker = pd.concat([lcoh_breakdown_tracker,pd.DataFrame(lcoh_breakdown_h2,index = [[re_cost_desc],[policy_desc],['no_storage']])])

                    pf_params=init_profast(model_params)
                    pf_tool= LCOH_Calc(model_params,cost_year,policy_desc,re_cost_desc)

                    sol,summary,price_breakdown,lcoh_breakdown = \
                    pf_tool.run_lcoh_full(copy.copy(pf_params),optimal_sizes,elec_eff_kWh_pr_kg,elec_cf,annual_hydrogen_kg,avg_stack_life_hrs,wind_frac,hydrogen_storage_capex_pr_kg,hydrogen_storage_opex_pr_kg)
                    
                    lcoh_full_tracker.append(sol['price'])
                    price_breakdown_tracker['{}-{}-{}'.format(re_cost_desc,policy_desc,storage_type)] = price_breakdown
                    lcoh_breakdown_tracker = pd.concat([lcoh_breakdown_tracker,pd.DataFrame(lcoh_breakdown,index = [[re_cost_desc],[policy_desc],[storage_type]])])

                    # pf_params=init_profast(model_params)
                    # pf_tool= LCOH_Calc(model_params,cost_year,policy_desc,re_cost_desc)

                    # sol_h2sto,summary_h2sto,price_breakdown_h2sto,lcoh_breakdown_h2sto =\
                    # pf_tool.run_lcoh2_storage(copy.copy(pf_params),elec_cf,optimal_sizes["electrolyzer_size_mw"],hydrogen_storage_size_kg,hydrogen_storage_capex_pr_kg,hydrogen_storage_opex_pr_kg)
                    #below is for testing
                    # optimal_sizes.update({"hydrogen_storage_size_kg":0})
                    # pf_tool.compressor_capex_pr_kWelec =0
                    # pf_tool.compressor_opex_pr_kWelec=0
                    # pf_params=init_profast(model_params)
                    # pf_tool= LCOH_Calc(model_params,cost_year,policy_desc,re_cost_desc)
                    # test_sol,test_summary,test_price_breakdown,test_lcoh_breakdown = \
                    # pf_tool.run_lcoh_full(copy.copy(pf_params),optimal_sizes,elec_eff_kWh_pr_kg,elec_cf,annual_hydrogen_kg,avg_stack_life_hrs,hydrogen_storage_capex_pr_kg,hydrogen_storage_opex_pr_kg)
                    # pd.Series(dict(zip(summary['Name'],summary['Amount'])))
                    # price_breakdown
                    # pd.Series(lcoh_breakdown)
                    
                    # # pd.DataFrame(dict(zip(summary['Type'],summary['Amount'])),index = summary['Name'])
                    # # h2_storage_lcoh = sol['lco']-test_sol['lco']
                    # # err = sol_h2sto['lco']-h2_storage_lcoh
                    # price_breakdown.loc[price_breakdown['Name']=='Compression']
                    # price_breakdown.loc[price_breakdown['Name']=='Hydrogen Storage']
                    # price_breakdown_h2sto.loc[price_breakdown_h2sto['Name']=='Compression']
                    # price_breakdown_h2sto.loc[price_breakdown_h2sto['Name']=='Hydrogen Storage']
                    # print('{}-{}-{}'.format(storage_type,re_cost_desc,policy_desc))
                    # print(err)
                    # if err>2.5:
                    #     []
        #finalize and save
        base_filename = '{}_{}-{}'.format(site_info['latitude'],site_info['longitude'],site_info['state'])
        optimal_sizes.update(H2_res)
        optimal_sizes.update(site_info)
        optimal_sizes.update({'wind_frac':wind_frac})
        ts_keys = ['Wind [kWh]','Solar [kWh]','Energy From Renewables [kWh]','Energy to Electrolyzer [kWh]','Hydrogen Hourly Production [kg/hour]']
        ts_vals = [wind_gen_kWh,solar_gen_kWh,energy_from_renewables,electrical_power_input_kWh,hydrogen_hourly_production]
        #save time series
        # pd.DataFrame(dict(zip(ts_keys,ts_vals))).to_csv(self.output_dir + 'performance/Timeseries_' + base_filename + '.csv')
        # pd.Series(optimal_sizes).to_csv(self.output_dir + 'performance/Summary_' + base_filename + '.csv')
        #save lcoh info
        # lcoh_breakdown_tracker.to_pickle(self.output_dir +'lcoh_results/LCOHBreakdown_' + base_filename + '.csv')
        # pd.Series(price_breakdown_tracker).to_pickle(self.output_dir +'lcoh_results/PriceBreakdown_' + base_filename)
        # lcoh_sum_keys = ['Renewables Case','Policy Case','Storage Case','LCOH (no storage)','LCOH (with storage)']
        lcoh_sum_keys = ['LCOH (no storage)','LCOH (with storage)']

        r_corr_coeff = self.calculate_correlation_coeff(wind_gen_kWh,solar_gen_kWh)
        optimal_sizes['correlation_coeff']=r_corr_coeff
        optimal_sizes['average_h2_kg_pr_hr'] = np.mean(hydrogen_hourly_production)
        optimal_sizes['max_h2_kg_pr_hr'] = np.max(hydrogen_hourly_production)
        
        optimal_sizes['Average kg-H2/day'] = H2_res['Life: Average Annual Hydrogen Produced [kg]']*(24/8760)
        optimal_sizes['Curtailed Power [MWh]'] = (np.sum(energy_from_renewables)-np.sum(electrical_power_input_kWh))/1000
        
        # lcoh_sum_vals = [renewable_cost_scenario,policy_scenario,storage_desc,lcoh_h2_tracker,lcoh_full_tracker]
        lcoh_sum_vals = [lcoh_h2_tracker,lcoh_full_tracker]
        # pd.DataFrame(dict(zip(lcoh_sum_keys,lcoh_sum_vals))).to_csv(self.output_dir +'lcoh_results/LCOHCaseSummary_' + base_filename + '.csv')
        all_outputs_to_save ={}
        summary_df = pd.DataFrame(dict(zip(lcoh_sum_keys,lcoh_sum_vals)))
        summary_df.index = [renewable_cost_scenario,policy_scenario,storage_desc]
        all_outputs_to_save['LCOHCaseSummary'] = summary_df
        all_outputs_to_save['PriceBreakdown'] = pd.Series(price_breakdown_tracker)
        all_outputs_to_save['LCOHBreakdown'] = lcoh_breakdown_tracker
        all_outputs_to_save['Timeseries'] = pd.DataFrame(dict(zip(ts_keys,ts_vals)))
        all_outputs_to_save['H2Res_Sizes'] = pd.Series(optimal_sizes)
        all_outputs_to_save['Hybrid_Plant']=pd.concat([pd.Series(dict(hybrid_plant.system_capacity_kw),name='System Capacity [kW]'),
        pd.Series(dict(hybrid_plant.capacity_factors),name='Capacity Factors'),
        pd.Series(dict(hybrid_plant.annual_energies),name='Annual Energy [kWh/year]')],axis=1)

        pd.Series(all_outputs_to_save).to_pickle(self.output_dir + 'lcoh_results/results_summary_' + base_filename)

        # ['LCOHCaseSummary','PriceBreakdown','LCOHBreakdown']
    def calculate_correlation_coeff(self,wind_gen_kWh,solar_gen_kWh):
        num = sum((wind_gen_kWh[i]-np.mean(wind_gen_kWh))*(solar_gen_kWh[i]-np.mean(solar_gen_kWh)) for i in range(len(wind_gen_kWh)))
        x_d = sum(((wind_gen_kWh[i]-np.mean(wind_gen_kWh))**2) for i in range(len(wind_gen_kWh)))
        y_d = sum(((solar_gen_kWh[i]-np.mean(solar_gen_kWh))**2) for i in range(len(wind_gen_kWh)))
        r = num/((x_d*y_d)**0.5)
        return r
    def run_battery(self,electrolyzer_size_mw,battery_size_mw,battery_hrs,energy_from_renewables):
        energy_upper_bound = electrolyzer_size_mw*1000
        energy_lower_bound = energy_upper_bound*0.1
        electrical_power_signal = np.where(energy_from_renewables >energy_upper_bound,energy_upper_bound,energy_from_renewables)
        from analysis.simple_dispatch import SimpleDispatch
        bat = SimpleDispatch()
        bat.Nt = len(energy_from_renewables)
        curtailed_power = np.where(energy_from_renewables >energy_upper_bound,energy_from_renewables -energy_upper_bound,0)
        sf_power = np.where(electrical_power_signal<energy_lower_bound,energy_lower_bound - electrical_power_signal,0)

        bat.curtailment = curtailed_power
        bat.shortfall = sf_power
        bat.battery_storage = battery_hrs*battery_size_mw*1000
        bat.charge_rate = battery_size_mw*1000
        bat.discharge_rate = battery_size_mw*1000
        battery_used, excess_energy, battery_SOC = bat.run()
        wind_solar_battery_energy = np.array(battery_used) + electrical_power_signal
        return wind_solar_battery_energy


        
    def init_electrolyzer_inputs(self,model_params):
        
        eol_eff_drop = model_params["electrolyzer"]["config"]["EOL_eff_drop"]
        user_defined_pem_param_dictionary = {
        "Modify BOL Eff": False,
        "BOL Eff [kWh/kg-H2]": [],
        "Modify EOL Degradation Value": True,
        "EOL Rated Efficiency Drop": eol_eff_drop,
        }
        return user_defined_pem_param_dictionary

    def run_electrolyezr_briefly(self,electrolyzer_size_mw,electrical_power_signal):
        from analysis.run_h2_PEM_mod import run_h2_PEM
        electrolyzer_direct_cost_kw_fake = 200 #only used for optimal control strategy
        num_clusters = electrolyzer_size_mw/self.stack_size_MW
        H2_Results, h2_kg_pr_hr= run_h2_PEM(electrical_power_signal, electrolyzer_size_mw,
                self.plant_life, num_clusters,  self.electrolysis_scale, 
                self.pem_control_type,electrolyzer_direct_cost_kw_fake, self.pem_input_dictionary,
                self.use_degradation_penalty, self.grid_connection_scenario,
                self.h2_reqd_kg_per_hr)
        return H2_Results,h2_kg_pr_hr
    def run_hopp(self,site_obj,wind_size_mw,solar_size_mw):
        #TODO: remove the policy thing from here, add to __init__ but check if used
        policy_keys = ['Wind ITC','Wind PTC','H2 PTC','Storage ITC']
        policy_vals = [0,0,0,0]
        hubht = self.wind_scenario["Tower Height"]
        rot_diam = self.wind_scenario['Rotor Diameter']
        num_turbs = np.ceil(wind_size_mw/self.turb_rating_mw)
        self.wind_scenario.update(dict(zip(policy_keys,policy_vals)))
        hpp = run_hybrid_plant(self.run_wind,self.run_solar,self.floris)
        technologies={
                'wind':{'num_turbines':num_turbs,
                'turbine_rating_kw':self.turb_rating_mw*1000,
                'hub_height': hubht,'rotor_diameter':rot_diam},
                'pv':{'system_capacity_kw':solar_size_mw*1000}
                }
        hybrid_plant=hpp.make_hybrid_plant(technologies,site_obj,self.wind_scenario)
        solar_gen_kWh = hpp.get_solar_generation(hybrid_plant)
        wind_gen_kWh = hpp.get_wind_generation(hybrid_plant)
        return solar_gen_kWh,wind_gen_kWh,hybrid_plant
    def run_optimizer(self,site_obj,model_params,pf_params,pf_tool_opt):
        # from optimization.gradient_opt_esg import simple_opt
        # from optimization.simple_param_sweep import param_sweep as simple_opt
        if model_params['optimization']['optimization_type'] == 'simple_param':
            # from optimization.param_sweep_esg import param_sweep as simple_opt
            from optimization.simple_param_sweep import param_sweep as simple_opt
        elif model_params['optimization']['optimization_type'] == 'simple_gradient':
            from optimization.param_sweep_esg import param_sweep as simple_opt

        elif model_params['optimization']['optimization_type'] == 'pyomo_opt':
            pass


        # from optimization.wrapper_code import run_optimization
        constraints = self.init_constraints(model_params)
        
        solar_gen_kWh_ref,wind_gen_kWh_ref,hybrid_plant = self.run_hopp(site_obj,constraints["ref_size_MW"]["wind"],constraints["ref_size_MW"]["solar"])
        
        res,all_res = simple_opt(wind_gen_kWh_ref,solar_gen_kWh_ref,constraints,pf_params,pf_tool_opt)
        if self.save_sweep_results: #TODO: finish this!
            # pd.Series(all_res).to_pickle(self.output_dir + 'sweep_results/all_sweep_results_{}-{}'.format(site_obj.lat,site_obj.lon))
            all_res.to_csv(self.output_dir + 'sweep_results/all_sweep_results_{}-{}'.format(site_obj.lat,site_obj.lon))
        size_idx = [i for i in list(res.index) if ('size' in i) or ('battery' in i)]
        return res[size_idx].to_dict()
        # res = run_optimization(hybrid_plant,constraints,T=8760)
    def post_process_optimal_results(self,res,constraints):
        #unused as of now
        S_wind_init = res["size_wind"]
        S_solar_init = res["size_solar"]
        S_elec_init = res["size_elec"]
        res["bat_used"] #battery discharged (MW)
        res["bat_SOC"] #battery energy level (MWh)
        res["p_charge_bat"] #plant power send to battery (MW)
        res["plant_to_dump"] #curtailed plant power (MW)
        res["plant_to_elec"] #power to electrolyzer from wind and solar
        res["elec_power"] #total power to electrolyzer
        pass

    def calc_hydrogen_storage_size(self,hydrogen_hourly_production,H2_demand = None):
        if H2_demand is None:
            H2_demand = np.mean(hydrogen_hourly_production)
        diff = hydrogen_hourly_production - H2_demand
        fake_soc = np.cumsum(diff)
        # if np.min(fake_soc)<0:
        #     hydrogen_storage_size_kg = np.max(fake_soc) + np.min(fake_soc)
        # else:
        hydrogen_storage_size_kg =np.max(fake_soc)- np.min(fake_soc)
        
        return np.abs(hydrogen_storage_size_kg)
    def init_wind(self,turb_rating_mw,turbine_filename):
        from tools.wind_farm_checker import check_wind
        wind_check = check_wind(self.main_dir,turb_rating_mw)
        #don't need wind_size_mw if using PySAM - only used if running FLORIS
        wind_size_mw = turb_rating_mw #I dont think this is used if using pysam
        wind_info = wind_check.run(wind_size_mw,self.floris,turbine_name = turbine_filename)
        wind_filename = wind_info['filename'] #double check
        wind_specs = [wind_info['Hub Height'],turb_rating_mw,\
        wind_filename, wind_info['Rotor Diameter']]
        wind_keys = ['Tower Height','Turbine Rating','Powercurve File','Rotor Diameter']
        wind_scenario = dict(zip(wind_keys,wind_specs))
        return wind_scenario #wind_info
        # {'filename':csv_filename,'Rotor Diameter':rot_diam,'Hub Height':hubht}
    def calc_desal_size(self,electrolyzer_size_mw,dt=3600):
        # water_density = 997 #[kg/m^3]
        
        rated_hydrogen_production_kg_hr = (electrolyzer_size_mw*1000)/self.elec_rated_eff_kWh_pr_kg#rated_kWh_per_kgH2
        rated_water_consumption_kg_hr = rated_hydrogen_production_kg_hr*self.kg_water_pr_kg_H2
        desal_sys_size_kg_sec = rated_water_consumption_kg_hr/dt
        return desal_sys_size_kg_sec
    def calc_hydrogen_storage_CapEx(self,hydrogen_storage_size_kg,storage_type,model_params):
        Sref = model_params["hydrogen_storage_cases"][storage_type]["base_capacity_kg"]
        Cref = model_params["hydrogen_storage_cases"][storage_type]["base_cost_USD_pr_kg"]
        f = model_params["hydrogen_storage_cases"][storage_type]["scaling_factor"]
        # h2_storage_opex=model_params["hydrogen_storage_cases"][storage_type]["opex_per_kg"]

        if hydrogen_storage_size_kg<Sref:
            hydrogen_storage_capex_pr_kg = Cref*(hydrogen_storage_size_kg/Sref)**(f-1)
        else:
            hydrogen_storage_capex_pr_kg = Cref
        return hydrogen_storage_capex_pr_kg
    def init_solar_constraints(self,model_params,constraints):
        #no minimum
        # model_params["solar"]["constraints"]
        # model_params["solar"]["config"]
        # model_params["solar"]["config"]["sim_default_losses"]

        constraints["max_size_MW"]["solar"] = model_params["solar"]["constraints"]["max_size_MW"]
        constraints["min_size_MW"]["solar"] = model_params["solar"]["constraints"]["min_size_MW"]
        constraints["ref_size_MW"]["solar"] = model_params["solar"]["constraints"]["ref_size_MW"]
        constraints["unit_size_MW"]["solar"] = model_params["solar"]["config"]["panel_rating_MW"]

        return constraints
    def init_battery_constraints(self,model_params,constraints):
        #no minimum
        max_hours = model_params["battery"]["constraints"]["max_nHours"]
        min_hours = model_params["battery"]["constraints"]["min_nHours"]
        charge_rate_MW = model_params["battery"]["constraints"]["max_chargeRate_MW"]
        # model_params["battery"]["config"]
        # model_params["battery"]["unit"]

        constraints["max_size_MW"]["battery"] = charge_rate_MW
        constraints["min_size_MW"]["battery"] = model_params["battery"]["constraints"]["min_chargeRate_MW"]
        constraints["ref_size_MW"]["battery"] = charge_rate_MW
        constraints["ref_size_MWh"]["battery"] = max_hours*charge_rate_MW
        constraints["unit_size_MW"]["battery"] = model_params["battery"]["config"]["unit_charge_rate_MW"]
        return constraints
    def init_wind_constraints(self,model_params,constraints):
        #min of 1 turb, max of 300 turbs
        # max_capac_MW = self.turb_rating_mw*model_params["wind"]["constraints"]["max_nTurbs"]
        # min_capac_MW = self.turb_rating_mw*model_params["wind"]["constraints"]["min_nTurbs"]
        # ref_capac_MW = self.turb_rating_mw*model_params["wind"]["constraints"]["ref_nTurbs"]
        # unit_size_MW = self.turb_rating_mw
        constraints["max_size_MW"]["wind"]= self.turb_rating_mw*model_params["turbine"]["constraints"]["max_nTurbs"]
        constraints["min_size_MW"]["wind"] = self.turb_rating_mw*model_params["turbine"]["constraints"]["min_nTurbs"]
        constraints["ref_size_MW"]["wind"] = self.turb_rating_mw*model_params["turbine"]["constraints"]["ref_nTurbs"]
        constraints["unit_size_MW"]["wind"] = model_params["turbine"]["config"]["turb_rating_mw"]
        # model_params["wind"]["config"]["turb_rating_mw"]
        # model_params["wind"]["config"]["sim_default_losses"]
        return constraints
    def init_electrolyzer_constraints(self,model_params,constraints):
        #min of 1 stack, max of 
        # model_params["electrolyzer"]["constraints"]
        stack_size_mw = model_params["electrolyzer"]["config"]["stack_size_MW"]

        constraints["max_size_MW"]["electrolyzer"] = stack_size_mw*model_params["electrolyzer"]["constraints"]["max_nStacks"]
        constraints["min_size_MW"]["electrolyzer"] = stack_size_mw*model_params["electrolyzer"]["constraints"]["min_nStacks"]
        constraints["ref_size_MW"]["electrolyzer"] = stack_size_mw*model_params["electrolyzer"]["constraints"]["ref_nStacks"]
        constraints["unit_size_MW"]["electrolyzer"] = stack_size_mw
        return constraints
    def init_constraints(self,model_params):
        constraints = {}
        constraints["max_size_MW"]={}
        constraints["min_size_MW"]={}
        constraints["ref_size_MW"]={}
        constraints["unit_size_MW"]={}
        constraints["ref_size_MWh"]={}

        constraints =self.init_electrolyzer_constraints(model_params,constraints)
        constraints =self.init_wind_constraints(model_params,constraints)
        constraints =self.init_solar_constraints(model_params,constraints)
        constraints =self.init_battery_constraints(model_params,constraints)


        return constraints
    def init_electrolyzer_efficiency(self):
        cluster_size_mw=1
        pem=PEM_H2_Clusters(cluster_size_mw, self.plant_life, user_defined_EOL_percent_eff_loss=False)
        rated_bol_eff = pem.output_dict['BOL Efficiency Curve Info']['Efficiency [kWh/kg]'].values[-1]
        # self.elec_rated_eff_kWh_pr_kg = rated_bol_eff
        # pass
        return rated_bol_eff
    def make_site(self,lat,lon,hub_ht,resource_year = 2013):
        sample_site = pd.read_pickle(self.main_dir + 'hybrid/sites/sample_site').to_dict()
        sample_site['lat']=lat
        sample_site['lon']=lon
        sample_site['year']= resource_year
        sample_site['no_wind'] = False
        sample_site['no_solar'] = False
        site = SiteInfo(sample_site,resource_dir = self.resource_directory,hub_height=hub_ht)

        return site

    