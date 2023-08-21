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
from scipy import interpolate
# from analysis.optimize_run_profast_for_hydrogen import opt_run_profast_for_hydrogen
from analysis.GS_v01_PF_for_H2 import LCOH_Calc
from analysis.GS_v01_set_PF_params import init_profast
# from analysis.GS_v01_PF_for_H2_Storage import LCOH_Calc
from analysis.PEM_H2_LT_electrolyzer_Clusters import PEM_H2_Clusters
# from analysis.opt_run_profast_for_h2_transmission import run_profast_for_h2_transmission
# from analysis.additional_cost_tools import hydrogen_storage_capacity_cost_calcs
from analysis.additional_cost_tools import hydrogen_storage_capacity_auto_calc_NEW,hydrogen_storage_cost_calc_NEW
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
        # cost_scenarios = model_params["simulation"]["cost_scenarios"]
        # storage_scenarios = model_params["simulation"]["storage_types"]
        storage_scenarios = ['buried_pipes','salt_cavern']
        cost_scenarios = ['Moderate','Advanced']
        
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
        solar_gen_kWh,wind_gen_kWh = self.run_hopp(site_obj,optimal_sizes["wind_size_mw"],optimal_sizes["solar_size_mw"])
        energy_from_renewables = solar_gen_kWh + wind_gen_kWh
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
        avg_stack_life_hrs=H2_res['Life: Stack Life [hrs]']
        
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
                    storage_desc.append(storage_type)
                    renewable_cost_scenario.append(re_cost_desc)
                    policy_scenario.append(policy_desc)

                    pf_params=init_profast(model_params)
                    pf_tool= LCOH_Calc(model_params,cost_year,policy_desc,re_cost_desc)
                    #run lcoh breakdown without hydrogen storage
                    sol_h2,summary_h2,price_breakdown_h2,lcoh_breakdown_h2 = \
                    pf_tool.run_lcoh_nostorage(copy.copy(pf_params),optimal_sizes,elec_eff_kWh_pr_kg,elec_cf,annual_hydrogen_kg,avg_stack_life_hrs)
                    price_breakdown_tracker['{}-{}-{}'.format(re_cost_desc,policy_desc,'no_storage')] = price_breakdown
                    lcoh_h2_tracker.append(sol_h2['price'])
                    lcoh_breakdown_tracker = pd.concat([lcoh_breakdown_tracker,pd.DataFrame(lcoh_breakdown_h2,index = [[re_cost_desc],[policy_desc],['no_storage']])])

                    pf_params=init_profast(model_params)
                    pf_tool= LCOH_Calc(model_params,cost_year,policy_desc,re_cost_desc)

                    sol,summary,price_breakdown,lcoh_breakdown = \
                    pf_tool.run_lcoh_full(copy.copy(pf_params),optimal_sizes,elec_eff_kWh_pr_kg,elec_cf,annual_hydrogen_kg,avg_stack_life_hrs,hydrogen_storage_capex_pr_kg,hydrogen_storage_opex_pr_kg)
                    
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
                    # print('{}-{}-{}'.format(storage_type,re_cost_desc,policy_desc,))
                    # print(err)
                    # if err>2.5:
                    #     []
        #finalize and save
        base_filename = '{}_{}-{}'.format(site_info['lat'],site_info['lon'],site_info['state'])
        optimal_sizes.update(H2_res)
        ts_keys = ['Wind [kWh]','Solar [kWh]','Energy From Renewables [kWh]','Energy to Electrolyzer [kWh]','Hydrogen Hourly Production [kg/hour]']
        ts_vals = [wind_gen_kWh,solar_gen_kWh,energy_from_renewables,electrical_power_input_kWh,hydrogen_hourly_production]
        #save time series
        # pd.DataFrame(dict(zip(ts_keys,ts_vals))).to_csv(self.output_dir + 'performance/Timeseries_' + base_filename + '.csv')
        # pd.Series(optimal_sizes).to_csv(self.output_dir + 'performance/Summary_' + base_filename + '.csv')
        #save lcoh info
        # lcoh_breakdown_tracker.to_pickle(self.output_dir +'lcoh_results/LCOHBreakdown_' + base_filename + '.csv')
        # pd.Series(price_breakdown_tracker).to_pickle(self.output_dir +'lcoh_results/PriceBreakdown_' + base_filename)
        lcoh_sum_keys = ['Renewables Case','Policy Case','Storage Case','LCOH (no storage)','LCOH (with storage)']
        lcoh_sum_vals = [renewable_cost_scenario,policy_scenario,storage_desc,lcoh_h2_tracker,lcoh_full_tracker]
        # pd.DataFrame(dict(zip(lcoh_sum_keys,lcoh_sum_vals))).to_csv(self.output_dir +'lcoh_results/LCOHCaseSummary_' + base_filename + '.csv')
        all_outputs_to_save ={}
        all_outputs_to_save['LCOHCaseSummary'] = pd.DataFrame(dict(zip(lcoh_sum_keys,lcoh_sum_vals)))
        all_outputs_to_save['PriceBreakdown'] = pd.Series(price_breakdown_tracker)
        all_outputs_to_save['LCOHBreakdown'] = lcoh_breakdown_tracker
        all_outputs_to_save['Timeseries'] = pd.DataFrame(dict(zip(ts_keys,ts_vals)))
        all_outputs_to_save['H2Res_Sizes'] = pd.Series(optimal_sizes)
        pd.Series(all_outputs_to_save).to_pickle(self.output_dir + 'lcoh_results/results_summary_' + base_filename)

        # ['LCOHCaseSummary','PriceBreakdown','LCOHBreakdown']

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
        return solar_gen_kWh,wind_gen_kWh
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
        
        solar_gen_kWh_ref,wind_gen_kWh_ref = self.run_hopp(site_obj,constraints["ref_size_MW"]["wind"],constraints["ref_size_MW"]["solar"])
        
        res,all_res = simple_opt(wind_gen_kWh_ref,solar_gen_kWh_ref,constraints,pf_params,pf_tool_opt)
        if self.save_sweep_results: #TODO: finish this!
            # pd.Series(all_res).to_pickle(self.output_dir + 'sweep_results/all_sweep_results_{}-{}'.format(site_obj.lat,site_obj.lon))
            all_res.to_csv(self.output_dir + 'sweep_results/all_sweep_results_{}-{}'.format(site_obj.lat,site_obj.lon))
        size_idx = [i for i in list(res.index) if ('size' in i) or ('battery' in i)]
        return res[size_idx].to_dict()
        # res = run_optimization(hybrid_plant,constraints,T=8760)
    def post_process_optimal_results(self,res,constraints):
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
        if np.min(fake_soc)<0:
            hydrogen_storage_size_kg = np.max(fake_soc) + np.min(fake_soc)
        else:
            hydrogen_storage_size_kg =np.max(fake_soc)- np.min(fake_soc)
        return hydrogen_storage_size_kg
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

    # def total_init(self):
    #     self.hpp=run_hybrid_plant(True,True,False)
    #     self.pem_tool=pem_tool('off-grid',self.end_of_life_eff_drop,100,self.plant_life)
    #     #100 is used as the placeholder electrolyzer CapEx - only used when using the optimal dispatch controller

    #     self.num_clusters = self.electrolyzer_capacity_MW/self.stack_size_MW
    #     self.pem_tool.init_opt_electrolyzer(self.electrolyzer_capacity_MW,self.num_clusters,self.annual_hydrogen_required_kg)
    #     self.init_scenario()
    #     self.hourly_h2_avg = self.annual_hydrogen_required_kg/8760
    #     self.pem_curt_thresh_kW = self.electrolyzer_capacity_MW*1000
    #     self.pem_sf_thresh_kW=self.pem_curt_thresh_kW *0.1
        
    #     self.solar_init_size_mw = 100
    #     self.solar_unit_size_mw = 10
    #     self.cost_years = [2025,2030,2035]
        
    #     self.pem_cost_cases = ['Mod 19','Mod 18']
    #     self.policy_cases = ['max','no policy']
    #     self.storage_types=['Salt cavern','Buried pipes']
    #     self.cost_years = [2025,2030,2035]
        # pem_bol_df = pd.read_pickle(self.input_info_dir + 'pem_1mw_bol_eff_pickle')
        # stack_input_power_kWh = self.stack_size_MW*pem_bol_df['Power Sent [kWh]'].values
        # stack_hydrogen_produced = self.stack_size_MW*pem_bol_df['H2 Produced'].values
        # p=np.insert(stack_input_power_kWh,0,0)
        # h=np.insert(stack_hydrogen_produced,0,0)
        # self.stack_kwh_to_h2 = interpolate.interp1d(p,h)

    
        
    # def wind_capac_2_lcoh(self,wind_power_kwh,wind_size_mw,cost_year,solar_power_kwh,solar_size_mw):
    #     self.electrolyzer_installed_cost=self.electrolyzer_installed_cost_opt.loc[cost_year].max()
    #     # wind_power_mult = np.array([1,1.5,2])
    #     wind_power_mult = np.array([1,1.15,1.3,1.45])
    #     wind_capac=wind_power_mult*wind_size_mw
    #     #solar_size_mw = 0
    #     policy_opt='no policy'
    #     storage_type='Salt caverns'
    #     # wind_lcoh=[]
    #     # wind_bat_lcoh=[]
    #     bat_size_mw=[]
    #     bat_size_hrs=[]
    #     all_lcoh=[]
    #     all_h2=[]
    #     wind_size=np.repeat(wind_capac,2)
    #     pv_size=[solar_size_mw]*(2*len(wind_power_mult))
    #     # wind_h2=[]
    #     # wind_bat_h2=[]
    #     # lcoh_wind_temp = 100
    #     # lcoh_wind_bat_temp = 100
    #     for wind_mult in wind_power_mult:
    #         wind_gen = (wind_power_kwh*wind_mult) + solar_power_kwh

    #         curtailed_wind = self.hpp.calc_curtailment(self.pem_curt_thresh_kW,wind_gen)
    #         wind_power = wind_gen - curtailed_wind
    #         H2_Results_wind,annual_h2_w = self.pem_tool.run_simple_opt_electrolyzer(wind_power)

    #         hydrogen_storage_capacity_kg,hydrogen_storage_duration_hr,hydrogen_storage_capacity_MWh_HHV=\
    #         hydrogen_storage_capacity_auto_calc_NEW(H2_Results_wind,self.electrolyzer_capacity_MW)
    #         hydrogen_storage_cost_USDprkg= hydrogen_storage_cost_calc_NEW(hydrogen_storage_capacity_MWh_HHV,storage_type)
    #         storage_data = (hydrogen_storage_capacity_kg,hydrogen_storage_cost_USDprkg)
    #         lcoh_wind = self.calc_lcoh(H2_Results_wind,policy_opt,storage_data,solar_size_mw,wind_size_mw*wind_mult,0,0,wind_power)
    #         # wind_h2.append(annual_h2_w)

    #         bat_size_mw.append(0)
    #         bat_size_hrs.append(0)
    #         all_lcoh.append(lcoh_wind)
    #         all_h2.append(annual_h2_w)

    #         bat_charge_rate_MW=10*np.round(np.ceil(np.max([np.max(curtailed_wind)/2,self.pem_sf_thresh_kW])/1000)/10)
    #         wind_bat_power,max_soc = self.try_battery(wind_gen,bat_charge_rate_MW,bat_storage_hrs=4)
    #         actual_bat_hrs = np.ceil(max_soc/(bat_charge_rate_MW*1000))
    #         H2_Results_bat,annual_h2_wb = self.pem_tool.run_simple_opt_electrolyzer(wind_bat_power)
    #         # wind_bat_h2.append(annual_h2_wb)
    #         hydrogen_storage_capacity_kg,hydrogen_storage_duration_hr,hydrogen_storage_capacity_MWh_HHV=\
    #         hydrogen_storage_capacity_auto_calc_NEW(H2_Results_bat,self.electrolyzer_capacity_MW)
    #         hydrogen_storage_cost_USDprkg= hydrogen_storage_cost_calc_NEW(hydrogen_storage_capacity_MWh_HHV,storage_type)
    #         storage_data = (hydrogen_storage_capacity_kg,hydrogen_storage_cost_USDprkg)
            
    #         lcoh_wind_bat=self.calc_lcoh(H2_Results_bat,policy_opt,storage_data,solar_size_mw,wind_size_mw*wind_mult,bat_charge_rate_MW,actual_bat_hrs,wind_bat_power)
    #         # wind_lcoh.append(lcoh_wind)
    #         # wind_bat_lcoh.append(lcoh_wind_bat)

    #         all_lcoh.append(lcoh_wind_bat)
    #         bat_size_mw.append(bat_charge_rate_MW)
    #         bat_size_hrs.append(actual_bat_hrs)
    #         all_h2.append(annual_h2_wb)
    #     return wind_size,pv_size,all_lcoh,all_h2,bat_size_mw,bat_size_hrs

    # def solar_capac_2_lcoh(self,solar_power_kwh,solar_size_mw,cost_year,wind_power,wind_size_mw):
    #     self.electrolyzer_installed_cost=self.electrolyzer_installed_cost_opt.loc[cost_year].max()
    #     pv_power_mult = np.array([2.5,5,7.5])
    #     pv_capac = pv_power_mult*solar_size_mw
        
    #     policy_opt='no policy'
    #     storage_type='Salt caverns'
    #     # wind_lcoh=[]
    #     # wind_bat_lcoh=[]
    #     bat_size_mw=[]
    #     bat_size_hrs=[]
    #     # pv_h2=[]
    #     # pv_bat_h2=[]
    #     all_lcoh=[]
    #     all_h2=[]
    #     pv_size=np.repeat(pv_capac,2)
    #     wind_size=[wind_size_mw]*(2*len(pv_power_mult))

    #     # lcoh_wind_temp = 100
    #     # lcoh_wind_bat_temp = 100
    #     for pv_mult in pv_power_mult:
    #         pv_gen = (solar_power_kwh*pv_mult) + wind_power

    #         curtailed_wind = self.hpp.calc_curtailment(self.pem_curt_thresh_kW,pv_gen)
    #         pv_power = pv_gen - curtailed_wind
    #         H2_Results_wind,annual_h2_pv = self.pem_tool.run_simple_opt_electrolyzer(pv_power)

    #         hydrogen_storage_capacity_kg,hydrogen_storage_duration_hr,hydrogen_storage_capacity_MWh_HHV=\
    #         hydrogen_storage_capacity_auto_calc_NEW(H2_Results_wind,self.electrolyzer_capacity_MW)
    #         hydrogen_storage_cost_USDprkg= hydrogen_storage_cost_calc_NEW(hydrogen_storage_capacity_MWh_HHV,storage_type)
    #         storage_data = (hydrogen_storage_capacity_kg,hydrogen_storage_cost_USDprkg)

    #         lcoh_wind = self.calc_lcoh(H2_Results_wind,policy_opt,storage_data,solar_size_mw*pv_mult,wind_size_mw,0,0,pv_power)
    #         # pv_h2.append(annual_h2_pv)

    #         all_lcoh.append(lcoh_wind)
    #         all_h2.append(annual_h2_pv)
    #         bat_size_mw.append(0)
    #         bat_size_hrs.append(0)

    #         bat_charge_rate_MW=10*np.round(np.ceil(np.max([np.max(curtailed_wind)/2,self.pem_sf_thresh_kW])/1000)/10)
    #         wind_bat_power,max_soc = self.try_battery(pv_gen,bat_charge_rate_MW,bat_storage_hrs=4)
    #         actual_bat_hrs = np.ceil(max_soc/(bat_charge_rate_MW*1000))
    #         H2_Results_bat,annual_h2_pvb = self.pem_tool.run_simple_opt_electrolyzer(wind_bat_power)
    #         # pv_bat_h2.append(annual_h2_pvb)
    #         hydrogen_storage_capacity_kg,hydrogen_storage_duration_hr,hydrogen_storage_capacity_MWh_HHV=\
    #         hydrogen_storage_capacity_auto_calc_NEW(H2_Results_bat,self.electrolyzer_capacity_MW)
    #         hydrogen_storage_cost_USDprkg= hydrogen_storage_cost_calc_NEW(hydrogen_storage_capacity_MWh_HHV,storage_type)
    #         storage_data = (hydrogen_storage_capacity_kg,hydrogen_storage_cost_USDprkg)


    #         lcoh_wind_bat=self.calc_lcoh(H2_Results_bat,policy_opt,storage_data,solar_size_mw*pv_mult,wind_size_mw,bat_charge_rate_MW,actual_bat_hrs,wind_bat_power)
    #         # wind_lcoh.append(lcoh_wind)
    #         # wind_bat_lcoh.append(lcoh_wind_bat)

    #         all_lcoh.append(lcoh_wind_bat)
    #         all_h2.append(annual_h2_pvb)

    #         bat_size_mw.append(bat_charge_rate_MW)
    #         bat_size_hrs.append(actual_bat_hrs)
    #     return wind_size,pv_size,all_lcoh,all_h2,bat_size_mw,bat_size_hrs
    #     # return pv_capac,wind_lcoh,wind_bat_lcoh,bat_size_mw,bat_size_hrs,pv_h2,pv_bat_h2

        
    # def run_optimizer_per_site(self,site_obj,wind_cf,cost_year):
    #     #site_obj is created with the SiteInfo function
    #     #wind_cf is the decimal of the wind capacity factor
    #     #cost year is the year we're running the parametric sweep for
    #     hpp=run_hybrid_plant(True,True,False)
    #     # losses_comp = 1.1
    #     if wind_cf<0.1:
    #         wind_cf = 0.3
    #     num_turbs_init = np.ceil((self.aep_MWh_reqd_min/(wind_cf*8760))/self.turb_rating_mw)
    #     if num_turbs_init>206:
    #         num_turbs_init=206
    #     wind_init_size_mw = num_turbs_init*self.turb_rating_mw
    #     technologies={
    #             'wind':{'num_turbines':num_turbs_init,
    #             'turbine_rating_kw':self.turb_rating_mw*1000,
    #             'hub_height': self.hubht,'rotor_diameter':self.rot_diam},
    #             'pv':{'system_capacity_kw':self.solar_init_size_mw*1000}
    #             }
    #     hybrid_plant=hpp.make_hybrid_plant(technologies,site_obj,self.scenario)
    #     init_wind_gen = hpp.get_wind_generation(hybrid_plant)
    #     init_solar_gen = hpp.get_solar_generation(hybrid_plant)
    #     # hybrid_plant.pv.capacity_factor/100
        
    #     self.renewable_plant_cost=self.init_renewables_costs(cost_year)
    #     # h2_per_mw_wind,h2_per_mw_solar=self.capac_2_h2_grad(init_wind_gen,init_solar_gen,wind_init_size_mw,self.solar_init_size_mw)
    #     # wind_capac,lcoh_wind,lcoh_wind_bat,bat_size_mw,bat_size_hrs,wind_h2,wind_bat_h2=\
    #     keys = ['Solar Size [MW]','Wind Size [MW]','Bat Size [MW]','Bat Hrs','LCOH','H2']

    #     wind_size,pv_size,all_lcoh,all_h2,bat_size_mw,bat_size_hrs=\
    #     self.wind_capac_2_lcoh(init_wind_gen,wind_init_size_mw,cost_year,init_solar_gen,self.solar_init_size_mw)
    #     w_vals = [pv_size,wind_size,bat_size_mw,bat_size_hrs,all_lcoh,all_h2]
    #     df=pd.DataFrame(dict(zip(keys,w_vals)))
    #     # wind_mult = wind_capac[np.argmin(lcoh_wind)]/wind_init_size_mw
    #     wind_mult = wind_size[np.argmin(all_lcoh)]/wind_init_size_mw
        

    #     # pv_capac,lcoh_pv,lcoh_pv_bat,bat_size_mw_with_pv,bat_size_hrs_with_pv,pv_h2,pv_bat_h2=\
    #     wind_size_pv,pv_size_pv,all_lcoh_pv,all_h2_pv,bat_size_mw_pv,bat_size_hrs_pv=\
    #     self.solar_capac_2_lcoh(init_solar_gen,self.solar_init_size_mw,cost_year,init_wind_gen*wind_mult,wind_size[np.argmin(all_lcoh)])
        
    #     # w_keys=['Solar Size [MW]','Wind Size [MW]','LCOH (wind)','H2 (wind)','LCOH (+ bat)','H2 (+bat)','Bat Size [MW]','Bat Hrs']
    #     # w_vals = [[self.solar_init_size_mw]*len(wind_capac),wind_capac,lcoh_wind,wind_h2,lcoh_wind_bat,wind_bat_h2,bat_size_mw,bat_size_hrs]

    #     # w_pv_keys=['Solar Size [MW]','Wind Size [MW]','LCOH (wind + pv)','H2 (wind + pv)','LCOH (+ bat)','H2 (+bat)','Bat Size [MW]','Bat Hrs']
    #     # w_pv_vals = [pv_capac,[wind_capac[np.argmin(lcoh_wind)]]*len(pv_capac),lcoh_pv,pv_h2,lcoh_pv_bat,pv_bat_h2,bat_size_mw_with_pv,bat_size_hrs_with_pv]
    #     # wpv_keys = ['Solar Size [MW]','Wind Size [MW]','Bat Size [MW]','Bat Hrs','LCOH','H2']
    #     wpv_vals = [pv_size_pv,wind_size_pv,bat_size_mw_pv,bat_size_hrs_pv,all_lcoh_pv,all_h2_pv]
    #     df=pd.concat([df,pd.DataFrame(dict(zip(keys,wpv_vals)))])
    #     df=df.reset_index(drop=True)
    #     return df
    #     # pd.DataFrame(dict(zip(wpv_keys,wpv_vals)))
        

        

    def try_battery(self,wind_solar_power,bat_charge_rate_MW,bat_storage_hrs):
        
        
        sf=self.hpp.calc_shortfall(self.pem_sf_thresh_kW,wind_solar_power)
        
        curt=self.hpp.calc_curtailment(self.pem_curt_thresh_kW,wind_solar_power)
       

        battery_dispatched_kWh,max_soc = self.hpp.general_simple_dispatch(bat_charge_rate_MW*1000*bat_storage_hrs,bat_charge_rate_MW*1000,sf,curt)
        re_plant_power = wind_solar_power-curt+battery_dispatched_kWh
        
        return re_plant_power,max_soc
    
    def init_scenario(self):
        # wind_filename = self.main_dir + 'input_params/lbw_6MW.csv' 
        wind_filename = self.main_dir + 'turbine_library/{}.csv'.format(self.turbine_name) 
        #assume we're using PySAM for wind farm simulation

        policy_keys = ['Wind ITC','Wind PTC','H2 PTC','Storage ITC']
        # if self.policy_desc == 'max':
        #     policy_vals = [0,0.03072,3.0,0.5]
        # else:
        policy_vals = [0,0,0,0] #NOTE: I don't think these policy vals are actually used
        #TODO: MAKE WIND SPECS FLEXIBLE
        wind_specs = [self.hubht,self.turb_rating_mw,\
        wind_filename, self.rot_diam]
        wind_keys = ['Tower Height','Turbine Rating','Powercurve File','Rotor Diameter']
        vals = policy_vals + wind_specs
        keys = policy_keys + wind_keys
        scenario = dict(zip(keys,vals))
        #TODO: add wind check here so it could change per site!
        self.scenario = scenario
    def run_best_case(self,site_obj,solar_size_mw,wind_size_mw,bat_size_mw,bat_hrs):
        num_turbs=np.ceil(wind_size_mw/self.turb_rating_mw)
        wind_act_size_mw = num_turbs*self.turb_rating_mw
        #these technologies only work if using pysam!
        technologies={
                'wind':{'num_turbines':num_turbs,
                'turbine_rating_kw':self.turb_rating_mw*1000,
                'hub_height': self.hubht,'rotor_diameter':self.rot_diam},
                'pv':{'system_capacity_kw':solar_size_mw*1000}
                }
        hpp=run_hybrid_plant(True,True,False)
        hybrid_plant=hpp.make_hybrid_plant(technologies,site_obj,self.scenario)
        wind_gen = hpp.get_wind_generation(hybrid_plant)
        solar_gen = hpp.get_solar_generation(hybrid_plant)
        
        wind_solar_power=wind_gen + solar_gen
        if bat_size_mw>0:
            energy_from_renewables,maxsoc = self.try_battery(wind_solar_power,bat_size_mw,bat_hrs)
        else:
            curt=self.hpp.calc_curtailment(self.pem_curt_thresh_kW,wind_solar_power)
            energy_from_renewables=wind_solar_power-curt

        # policy_cases = ['max','no policy']
        # storage_types=['Salt cavern','Buried pipes']
        # H2_Results,annual_h2 = self.pem_tool.run_simple_opt_electrolyzer(energy_from_renewables)
        
        H2_Results,h2_df_tot = self.pem_tool.run_full_opt_electrolyzer(energy_from_renewables)
        hydrogen_storage_capacity_kg,hydrogen_storage_duration_hr,hydrogen_storage_capacity_MWh_HHV=\
            hydrogen_storage_capacity_auto_calc_NEW(H2_Results,self.electrolyzer_capacity_MW)
        h2_sto_keys=['H2 Storage [kg]','H2 Storage [hour]','H2 Storage MWh_HHV']
        h2_sto_data=[hydrogen_storage_capacity_kg,hydrogen_storage_duration_hr,hydrogen_storage_capacity_MWh_HHV]
        
        # lcoht,h2_tran_info=self.calc_hydrogen_transmission_cost(H2_Results)
        lcoh_tracker=[]
        lcoh_breakdowns = pd.DataFrame()
        lcoh_names = []
        h2_transmission_df=pd.DataFrame()
        for storage_type in self.storage_types:
            h2_sto_keys.append(storage_type + ': H2 Storage CapEx [$/kg]')
            hydrogen_storage_cost_USDprkg= hydrogen_storage_cost_calc_NEW(hydrogen_storage_capacity_MWh_HHV,storage_type)
            h2_sto_data.append(hydrogen_storage_cost_USDprkg)
            
            storage_data = (hydrogen_storage_capacity_kg,hydrogen_storage_cost_USDprkg)
            for cost_year in self.cost_years:
                self.renewable_plant_cost=self.init_renewables_costs(cost_year)
                lcoht,h2_tran_info=self.calc_hydrogen_transmission_cost(H2_Results)
                if storage_type==self.storage_types[0]:
                    h2_tran_info.name=cost_year
                    h2_transmission_df=pd.concat([h2_transmission_df,h2_tran_info],axis=1)
                for pem_cost_case in self.pem_cost_cases:
                    self.electrolyzer_installed_cost=self.electrolyzer_installed_cost_opt.loc[cost_year][pem_cost_case]
                            
                    for policy in self.policy_cases:
                        lcoh_desc = '{}_{}_{}_{}'.format(cost_year,pem_cost_case.replace(' ',''),storage_type.replace(' ',''),policy.replace(' ',''))
                        lcoh_init,lcoh_details=self.calc_lcoh(H2_Results,policy,storage_data,solar_size_mw,wind_act_size_mw,bat_size_mw,bat_hrs,energy_from_renewables,return_details=True)
                        lcoh = lcoh_init + lcoht
                        lcoh_tracker.append(lcoh)
                        lcoh_names.append(lcoh_desc)
                        lcoh_details.name = lcoh_desc
                        lcoh_breakdowns=pd.concat([lcoh_breakdowns,lcoh_details],axis=1)
        
        desc_keys = ['Wind [MW]','Solar [MW]','Battery [MW]','Battery [Hrs]','Wind CF','PV CF','PEM CF']
        desc_vals = [wind_act_size_mw,solar_size_mw,bat_size_mw,bat_hrs,hybrid_plant.wind.capacity_factor/100,hybrid_plant.pv.capacity_factor/100,H2_Results['cap_factor']]
        ts_data = pd.DataFrame({'Wind [kWh]':wind_gen,'Solar [kWh]':solar_gen,'Total Energy':energy_from_renewables})
        # h2_sto_keys=['H2 Storage [kg]','H2 Storage [hour]','H2 Storage Cost [$/kg]']
        # h2_sto_data=[hydrogen_storage_capacity_kg,hydrogen_storage_duration_hr,hydrogen_storage_cost_USDprkg]
        # extra_data = {'Wind CF':hybrid_plant.wind.capacity_factor/100,'PV CF':hybrid_plant.pv.capacity_factor,}
        all_info = {'Case Desc':dict(zip(desc_keys,desc_vals)),'H2 Transmission Info':h2_transmission_df,'H2 Storage Info':dict(zip(h2_sto_keys,h2_sto_data)),'LCOH Breakdowns':lcoh_breakdowns,'H2 Results':H2_Results,'H2 Tot':h2_df_tot,'LCOH Per Case':pd.Series(dict(zip(lcoh_names,lcoh_tracker))),'Time Series':ts_data}
        
        # hybrid_plant.pv.capacity_factor

        return pd.Series(all_info)

    def make_site(self,lat,lon,hub_ht,resource_year = 2013):
        sample_site = pd.read_pickle(self.main_dir + 'hybrid/sites/sample_site').to_dict()
        sample_site['lat']=lat
        sample_site['lon']=lon
        sample_site['year']= resource_year
        sample_site['no_wind'] = False
        sample_site['no_solar'] = False
        site = SiteInfo(sample_site,resource_dir = self.resource_directory,hub_height=hub_ht)

        return site
        
    def run_single_site(self,site_info,results_subdir):
        self.save_dir = self.main_dir + 'results/' + results_subdir + '/'
        opt_cost_year = 2030
        site_desc = '{}_ID{}_{}_{}'.format(site_info['State'].replace(' ',''),site_info['site_id'],site_info['latitude'],site_info['longitude'])
        site_obj = self.make_site(site_info['latitude'],site_info['longitude'],self.hubht)
        df_init = self.run_optimizer_per_site(site_obj,site_info['capacity_factor'],opt_cost_year)
        lcoh_idx=df_init['LCOH'].idxmin()
        all_info=self.run_best_case(site_obj,df_init['Solar Size [MW]'].iloc[lcoh_idx],df_init['Wind Size [MW]'].iloc[lcoh_idx],df_init['Bat Size [MW]'].iloc[lcoh_idx],df_init['Bat Hrs'].iloc[lcoh_idx])

        df_init.to_pickle(self.save_dir + 'sweep_results/' + site_desc + '_{}'.format(opt_cost_year))
        all_info.to_pickle(self.save_dir + 'lcoh_results/' + site_desc)
        []




    def run_all(self,site_df,results_subdir):
        self.save_dir = self.main_dir + 'results/' + results_subdir + '/'
        opt_cost_year = 2030
        
        
        for i in range(len(site_df)):
            # site_desc='Test_{}'.format(site_df.iloc[i]['Site ID'])
            # site_desc = '{}_ID{}_{}_{}'.format(site_df.iloc[i]['State'].replace(' ',''),site_df.iloc[i]['Site ID'],site_df.iloc[i]['Lat'],site_df.iloc[i]['Lon'])
            site_desc = '{}_ID{}_{}_{}'.format(site_df.iloc[i]['State'].replace(' ',''),site_df.iloc[i]['site_id'],site_df.iloc[i]['latitude'],site_df.iloc[i]['longitude'])
            # self.renewable_plant_cost=self.init_renewables_costs(cost_year)
            site_obj = self.make_site(site_df.iloc[i]['latitude'],site_df.iloc[i]['longitude'],self.hubht)
            
            df_init = self.run_optimizer_per_site(site_obj,site_df.iloc[i]['capacity_factor'],opt_cost_year)
            
            lcoh_idx=df_init['LCOH'].idxmin()
            # df_init['Solar Size [MW]'].iloc[lcoh_idx]
            # df_init['Wind Size [MW]'].iloc[lcoh_idx]
            # df_init['Bat Size [MW]'].iloc[lcoh_idx]
            # df_init['Bat Hrs [MW]'].iloc[lcoh_idx]
            # all_info=self.run_best_case(site_df.iloc[i]['Site Obj'],df_init['Solar Size [MW]'].iloc[lcoh_idx],df_init['Wind Size [MW]'].iloc[lcoh_idx],df_init['Bat Size [MW]'].iloc[lcoh_idx],df_init['Bat Hrs'].iloc[lcoh_idx])
            all_info=self.run_best_case(site_obj,df_init['Solar Size [MW]'].iloc[lcoh_idx],df_init['Wind Size [MW]'].iloc[lcoh_idx],df_init['Bat Size [MW]'].iloc[lcoh_idx],df_init['Bat Hrs'].iloc[lcoh_idx])
            # keys = ['Solar Size [MW]','Wind Size [MW]','Bat Size [MW]','Bat Hrs','LCOH','H2']
            df_init.to_pickle(self.save_dir + 'sweep_results/' + site_desc + '_{}'.format(opt_cost_year))
            all_info.to_pickle(self.save_dir + 'lcoh_results/' + site_desc)
            []

    
    def calc_hydrogen_transmission_cost(self,H2_Results):
        
        pipeline_length_km = 50
        enduse_capacity_factor = 0.9
        #^used if "after"
        before_after_storage = 'after' #I think this is cheaper than before
        max_hydrogen_production_rate_kg_hr = np.max(H2_Results['hydrogen_hourly_production'])
        #^used if "before"
        max_hydrogen_delivery_rate_kg_hr  = np.mean(H2_Results['hydrogen_hourly_production'])
        #^used if "after"
        
        electrolyzer_capacity_factor = H2_Results['cap_factor']

        h2_transmission_economics_from_profast,h2_transmission_price_breakdown,h2_df=\
        run_profast_for_h2_transmission(max_hydrogen_production_rate_kg_hr,\
        max_hydrogen_delivery_rate_kg_hr,pipeline_length_km,electrolyzer_capacity_factor,\
        enduse_capacity_factor,before_after_storage,self.plant_life,self.elec_price)

        
        h2_transmission_price = h2_transmission_economics_from_profast['price']

        
        keys=['LCOHT [$/kg-H2]','H2 Tran Breakdown','H2 Transmission Details']
        vals=[h2_transmission_price,h2_transmission_price_breakdown,h2_df]
        h2_tran_info = pd.Series(dict(zip(keys,vals)),name='H2 Transmission Info')
        lcoht=h2_transmission_price
        return lcoht,h2_tran_info


    
    def calc_lcoh(self,H2_Results,policy_option,storage_type,solar_size_mw,wind_size_mw,battery_size_mw,battery_hrs,energy_from_renewables_kWh,return_details=False):
        # self.electrolyzer_installed_cost_opt #loop it
        hydrogen_storage_capacity_kg,hydrogen_storage_cost_USDprkg=storage_type
        # hydrogen_storage_capacity_kg,hydrogen_storage_duration_hr,hydrogen_storage_cost_USDprkg\
        #     =hydrogen_storage_capacity_cost_calcs(H2_Results,self.electrolyzer_capacity_MW,storage_type)
        #https://www.osti.gov/biblio/1975260: Figure 4 average of about 4$/kgal
        water_cost = 0.004 #$/gal
        
        h2_solution,lcoh_breakdown,capex_df=\
            opt_run_profast_for_hydrogen(self.electrolyzer_installed_cost,\
                            self.electrolyzer_capacity_MW,H2_Results,\
                            hydrogen_storage_capacity_kg,hydrogen_storage_cost_USDprkg,\
                            self.renewable_plant_cost,energy_from_renewables_kWh,\
                            policy_option,self.plant_life,water_cost, \
                            wind_size_mw,solar_size_mw,battery_size_mw,battery_hrs)
        
        lcoh_init = h2_solution['price']
        
        if return_details:
            
            gen_cost_keys = ['LCOH (no transmission)','LCOH Breakdown','CapEx Per Component']
            gen_cost_vals = [lcoh_init,lcoh_breakdown,capex_df]

            lcoh_info = pd.Series(dict(zip(gen_cost_keys,gen_cost_vals)))
            return lcoh_init,lcoh_info
        
        else:
            return lcoh_init
        


    
#from elenya_write_outputs import esg_write_outputs_ProFAST
    def write_outputs(self):
        print("writing outputs...")
        pass



if __name__ == "__main__":
    start_indx = 0
    api_call_day=0
    parent_path = os.path.abspath('') + '/'
    main_dir = parent_path + 'NATIONAL_SWEEP/'
    default_input_file = main_dir + 'sweep_defaults.csv'
    model_params = pd.read_csv(default_input_file,index_col = 'Variable')
    sites = pd.read_pickle(main_dir + 'Sites7k_Alabama_day0')

    year=2030
    site_list = []
    policy_opt = ['no policy']
    #def __init__(self,model_params,year,site_list,policy_opt)
    #
    pem_cost_cases = ['Mod 19','Mod 18']
    policy_cases = ['max','no policy']
    sweep = opt_national_sweep(model_params)
    results_subdir ='100mHubHt_Try01'
    sites['cap_fac']=0.4
    sweep.run_all(sites,115,results_subdir)

    []
    site_obj=sites['Site Obj'].iloc[0]
    site_id = sites['Site ID'].iloc[0]
    locs = pd.read_csv(main_dir + 'wtk_simple_sites_7kadj_onshore_enoughArea.csv',index_col = 'site_id')
    wind_cf = locs.loc[site_id]['capacity_factor']
    sweep.run_optimizer_per_site(site_obj,wind_cf)
    sweep.quick_hybrid_plant_check(sites['Site Obj'].iloc[0])
    sweep.init_pem_info()
    
    []










