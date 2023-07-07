import os
import sys
#sys.path.append('')
from dotenv import load_dotenv
import pandas as pd
from hybrid.sites.site_info  import SiteInfo

import numpy as np

import warnings
# from pathlib import Path
# import time
from analysis.hybrid_plant_tools import run_hybrid_plant
from scipy import interpolate
from analysis.optimize_run_profast_for_hydrogen import opt_run_profast_for_hydrogen
# from analysis.run_profast_for_h2_transmission import run_profast_for_h2_transmission
from analysis.opt_run_profast_for_h2_transmission import run_profast_for_h2_transmission
# from analysis.additional_cost_tools import hydrogen_storage_capacity_cost_calcs
from analysis.additional_cost_tools import hydrogen_storage_capacity_auto_calc_NEW,hydrogen_storage_cost_calc_NEW
from analysis.electrolyzer_tools import electrolyzer_tools as pem_tool
from tools.floris_2_csv import csv_from_floris
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
        # self.main_dir= os.path.abspath('') + '/'
        self.input_info_dir = self.main_dir + 'input_info/'
        # self.input_info_dir = self.main_dir + 'input_params/'
        
        
        pem_cost = pd.read_csv(self.input_info_dir + 'pem_installed_capex.csv',index_col='Year')
        
        self.electrolyzer_installed_cost_opt=pem_cost[pem_cost_cases]
        self.resource_directory = self.main_dir + 'resource_files/'
        
        # default_input_file = self.main_dir + 'sweep_defaults.csv'
        self.model_params = model_params#pd.read_csv(default_input_file,index_col = 'Variable')

        # DEFAULTS #
        self.resource_year = 2013
        self.annual_hydrogen_required_kg = 66000*1000 #metric ton-> kg
        self.end_of_life_eff_drop = 10 #for electrolyzer
        
        self.plant_life = 30
        #66,000 metric tons of hydrogen per year.
        # self.electrolyzer_capacity_MW = 720 #800
        target_electrolyzer_cf = 0.6
        hourly_kgh2_target = self.annual_hydrogen_required_kg/(target_electrolyzer_cf*8760)
        required_electrolyzer_capacity_MW = hourly_kgh2_target/18.11

        # self.annual_hydrogen_required_kg = self.target_electrolyzer_cf*8760*(self.electrolyzer_capacity_MW*18.11)

        self.stack_size_MW = 40
        num_clusters = np.round(required_electrolyzer_capacity_MW/self.stack_size_MW)
        self.electrolyzer_capacity_MW = self.stack_size_MW*num_clusters
        self.aep_MWh_reqd_min=self.annual_hydrogen_required_kg*54.8/1000
        # self.aep_MWh_reqd_min = 3073211
        # self.aep_MWh_reqd_max = self.aep_MWh_reqd_min*1.3
        #20 stacks rated at 40
        
        
        self.hubht = model_params.loc['Hub Height'].values[0]
        self.rot_diam = model_params.loc['Rotor Diameter'].values[0]
        self.turb_rating_mw = model_params.loc['Turbine Rating'].values[0]
        self.turbine_name = 'lbw_6MW'
        self.total_init()

        #csv_from_floris(self.main_dir + 'input_params/','lbw_6MW')
        
        
    def init_renewables_costs(self,cost_year):
        #Grid cost only for h2 transmission
        #https://www.eia.gov/electricity/state/
        ref_year = 2021
        average_grid_retail_rate = 11.1/100 #$/kWh for 2021
        price_inc_per_year = 1.2/1000 # $[$/kWh/year] from $1/MWh/year average
        self.elec_price=(price_inc_per_year*(cost_year-ref_year)) + average_grid_retail_rate

        #battery storage costs
        st_xl=pd.read_csv(self.input_info_dir + 'battery_storage_costs.csv',index_col=0)
        storage_costs=st_xl[str(cost_year)]
        
        storage_cost_kwh=storage_costs['Battery Energy Capital Cost ($/kWh)']
        storage_cost_kw=storage_costs['Battery Power Capital Cost ($/kW)'] 
        #solar costs
        pv_capex = self.model_params.loc['{} PV base installed cost'.format(cost_year)].values[0]
        pv_opex = self.model_params.loc['{} PV OpEx'.format(cost_year)].values[0]
        pv_fcr=self.model_params.loc['PV FCR (all years)'].values[0]

        #wind costs
        wind_capex = self.model_params.loc['{} CapEx'.format(cost_year)].values[0]
        wind_opex = self.model_params.loc['{} OpEx ($/kw-yr)'.format(cost_year)].values[0]
        wind_fcr = self.model_params.loc['FCR (all years)'].values[0]
        renewable_plant_cost={}
        renewable_plant_cost['wind']={
            'o&m_per_kw':wind_opex,
            'capex_per_kw':wind_capex,
            'FCR':wind_fcr}
        renewable_plant_cost['pv']={
            'o&m_per_kw':pv_opex,
            'capex_per_kw':pv_capex,
            'FCR':pv_fcr}
        renewable_plant_cost['battery']={
                    'capex_per_kw':storage_cost_kw,
                    'capex_per_kwh':storage_cost_kwh,
                    'o&m_percent':0.025,
                    } 
        return renewable_plant_cost

    def total_init(self):
        self.hpp=run_hybrid_plant(True,True,False)
        self.pem_tool=pem_tool('off-grid',self.end_of_life_eff_drop,100,self.plant_life)
        #100 is used as the placeholder electrolyzer CapEx - only used when using the optimal dispatch controller

        self.num_clusters = self.electrolyzer_capacity_MW/self.stack_size_MW
        self.pem_tool.init_opt_electrolyzer(self.electrolyzer_capacity_MW,self.num_clusters,self.annual_hydrogen_required_kg)
        self.init_scenario()
        self.hourly_h2_avg = self.annual_hydrogen_required_kg/8760
        self.pem_curt_thresh_kW = self.electrolyzer_capacity_MW*1000
        self.pem_sf_thresh_kW=self.pem_curt_thresh_kW *0.1
        
        self.solar_init_size_mw = 100
        self.solar_unit_size_mw = 10
        self.cost_years = [2025,2030,2035]
        
        self.pem_cost_cases = ['Mod 19','Mod 18']
        self.policy_cases = ['max','no policy']
        self.storage_types=['Salt cavern','Buried pipes']
        self.cost_years = [2025,2030,2035]
        # pem_bol_df = pd.read_pickle(self.input_info_dir + 'pem_1mw_bol_eff_pickle')
        # stack_input_power_kWh = self.stack_size_MW*pem_bol_df['Power Sent [kWh]'].values
        # stack_hydrogen_produced = self.stack_size_MW*pem_bol_df['H2 Produced'].values
        # p=np.insert(stack_input_power_kWh,0,0)
        # h=np.insert(stack_hydrogen_produced,0,0)
        # self.stack_kwh_to_h2 = interpolate.interp1d(p,h)

    
        
    def wind_capac_2_lcoh(self,wind_power_kwh,wind_size_mw,cost_year,solar_power_kwh,solar_size_mw):
        self.electrolyzer_installed_cost=self.electrolyzer_installed_cost_opt.loc[cost_year].max()
        # wind_power_mult = np.array([1,1.5,2])
        wind_power_mult = np.array([1,1.15,1.3,1.45])
        wind_capac=wind_power_mult*wind_size_mw
        #solar_size_mw = 0
        policy_opt='no policy'
        storage_type='Salt caverns'
        # wind_lcoh=[]
        # wind_bat_lcoh=[]
        bat_size_mw=[]
        bat_size_hrs=[]
        all_lcoh=[]
        all_h2=[]
        wind_size=np.repeat(wind_capac,2)
        pv_size=[solar_size_mw]*(2*len(wind_power_mult))
        # wind_h2=[]
        # wind_bat_h2=[]
        # lcoh_wind_temp = 100
        # lcoh_wind_bat_temp = 100
        for wind_mult in wind_power_mult:
            wind_gen = (wind_power_kwh*wind_mult) + solar_power_kwh

            curtailed_wind = self.hpp.calc_curtailment(self.pem_curt_thresh_kW,wind_gen)
            wind_power = wind_gen - curtailed_wind
            H2_Results_wind,annual_h2_w = self.pem_tool.run_simple_opt_electrolyzer(wind_power)

            hydrogen_storage_capacity_kg,hydrogen_storage_duration_hr,hydrogen_storage_capacity_MWh_HHV=\
            hydrogen_storage_capacity_auto_calc_NEW(H2_Results_wind,self.electrolyzer_capacity_MW)
            hydrogen_storage_cost_USDprkg= hydrogen_storage_cost_calc_NEW(hydrogen_storage_capacity_MWh_HHV,storage_type)
            storage_data = (hydrogen_storage_capacity_kg,hydrogen_storage_cost_USDprkg)
            lcoh_wind = self.calc_lcoh(H2_Results_wind,policy_opt,storage_data,solar_size_mw,wind_size_mw*wind_mult,0,0,wind_power)
            # wind_h2.append(annual_h2_w)

            bat_size_mw.append(0)
            bat_size_hrs.append(0)
            all_lcoh.append(lcoh_wind)
            all_h2.append(annual_h2_w)

            bat_charge_rate_MW=10*np.round(np.ceil(np.max([np.max(curtailed_wind)/2,self.pem_sf_thresh_kW])/1000)/10)
            wind_bat_power,max_soc = self.try_battery(wind_gen,bat_charge_rate_MW,bat_storage_hrs=4)
            actual_bat_hrs = np.ceil(max_soc/(bat_charge_rate_MW*1000))
            H2_Results_bat,annual_h2_wb = self.pem_tool.run_simple_opt_electrolyzer(wind_bat_power)
            # wind_bat_h2.append(annual_h2_wb)
            hydrogen_storage_capacity_kg,hydrogen_storage_duration_hr,hydrogen_storage_capacity_MWh_HHV=\
            hydrogen_storage_capacity_auto_calc_NEW(H2_Results_bat,self.electrolyzer_capacity_MW)
            hydrogen_storage_cost_USDprkg= hydrogen_storage_cost_calc_NEW(hydrogen_storage_capacity_MWh_HHV,storage_type)
            storage_data = (hydrogen_storage_capacity_kg,hydrogen_storage_cost_USDprkg)
            
            lcoh_wind_bat=self.calc_lcoh(H2_Results_bat,policy_opt,storage_data,solar_size_mw,wind_size_mw*wind_mult,bat_charge_rate_MW,actual_bat_hrs,wind_bat_power)
            # wind_lcoh.append(lcoh_wind)
            # wind_bat_lcoh.append(lcoh_wind_bat)

            all_lcoh.append(lcoh_wind_bat)
            bat_size_mw.append(bat_charge_rate_MW)
            bat_size_hrs.append(actual_bat_hrs)
            all_h2.append(annual_h2_wb)
        return wind_size,pv_size,all_lcoh,all_h2,bat_size_mw,bat_size_hrs

    def solar_capac_2_lcoh(self,solar_power_kwh,solar_size_mw,cost_year,wind_power,wind_size_mw):
        self.electrolyzer_installed_cost=self.electrolyzer_installed_cost_opt.loc[cost_year].max()
        pv_power_mult = np.array([2.5,5,7.5])
        pv_capac = pv_power_mult*solar_size_mw
        
        policy_opt='no policy'
        storage_type='Salt caverns'
        # wind_lcoh=[]
        # wind_bat_lcoh=[]
        bat_size_mw=[]
        bat_size_hrs=[]
        # pv_h2=[]
        # pv_bat_h2=[]
        all_lcoh=[]
        all_h2=[]
        pv_size=np.repeat(pv_capac,2)
        wind_size=[wind_size_mw]*(2*len(pv_power_mult))

        # lcoh_wind_temp = 100
        # lcoh_wind_bat_temp = 100
        for pv_mult in pv_power_mult:
            pv_gen = (solar_power_kwh*pv_mult) + wind_power

            curtailed_wind = self.hpp.calc_curtailment(self.pem_curt_thresh_kW,pv_gen)
            pv_power = pv_gen - curtailed_wind
            H2_Results_wind,annual_h2_pv = self.pem_tool.run_simple_opt_electrolyzer(pv_power)

            hydrogen_storage_capacity_kg,hydrogen_storage_duration_hr,hydrogen_storage_capacity_MWh_HHV=\
            hydrogen_storage_capacity_auto_calc_NEW(H2_Results_wind,self.electrolyzer_capacity_MW)
            hydrogen_storage_cost_USDprkg= hydrogen_storage_cost_calc_NEW(hydrogen_storage_capacity_MWh_HHV,storage_type)
            storage_data = (hydrogen_storage_capacity_kg,hydrogen_storage_cost_USDprkg)

            lcoh_wind = self.calc_lcoh(H2_Results_wind,policy_opt,storage_data,solar_size_mw*pv_mult,wind_size_mw,0,0,pv_power)
            # pv_h2.append(annual_h2_pv)

            all_lcoh.append(lcoh_wind)
            all_h2.append(annual_h2_pv)
            bat_size_mw.append(0)
            bat_size_hrs.append(0)

            bat_charge_rate_MW=10*np.round(np.ceil(np.max([np.max(curtailed_wind)/2,self.pem_sf_thresh_kW])/1000)/10)
            wind_bat_power,max_soc = self.try_battery(pv_gen,bat_charge_rate_MW,bat_storage_hrs=4)
            actual_bat_hrs = np.ceil(max_soc/(bat_charge_rate_MW*1000))
            H2_Results_bat,annual_h2_pvb = self.pem_tool.run_simple_opt_electrolyzer(wind_bat_power)
            # pv_bat_h2.append(annual_h2_pvb)
            hydrogen_storage_capacity_kg,hydrogen_storage_duration_hr,hydrogen_storage_capacity_MWh_HHV=\
            hydrogen_storage_capacity_auto_calc_NEW(H2_Results_bat,self.electrolyzer_capacity_MW)
            hydrogen_storage_cost_USDprkg= hydrogen_storage_cost_calc_NEW(hydrogen_storage_capacity_MWh_HHV,storage_type)
            storage_data = (hydrogen_storage_capacity_kg,hydrogen_storage_cost_USDprkg)


            lcoh_wind_bat=self.calc_lcoh(H2_Results_bat,policy_opt,storage_data,solar_size_mw*pv_mult,wind_size_mw,bat_charge_rate_MW,actual_bat_hrs,wind_bat_power)
            # wind_lcoh.append(lcoh_wind)
            # wind_bat_lcoh.append(lcoh_wind_bat)

            all_lcoh.append(lcoh_wind_bat)
            all_h2.append(annual_h2_pvb)

            bat_size_mw.append(bat_charge_rate_MW)
            bat_size_hrs.append(actual_bat_hrs)
        return wind_size,pv_size,all_lcoh,all_h2,bat_size_mw,bat_size_hrs
        # return pv_capac,wind_lcoh,wind_bat_lcoh,bat_size_mw,bat_size_hrs,pv_h2,pv_bat_h2

        
    def run_optimizer_per_site(self,site_obj,wind_cf,cost_year):
        #site_obj is created with the SiteInfo function
        #wind_cf is the decimal of the wind capacity factor
        #cost year is the year we're running the parametric sweep for
        hpp=run_hybrid_plant(True,True,False)
        # losses_comp = 1.1
        if wind_cf<0.1:
            wind_cf = 0.3
        num_turbs_init = np.ceil((self.aep_MWh_reqd_min/(wind_cf*8760))/self.turb_rating_mw)
        if num_turbs_init>206:
            num_turbs_init=206
        wind_init_size_mw = num_turbs_init*self.turb_rating_mw
        technologies={
                'wind':{'num_turbines':num_turbs_init,
                'turbine_rating_kw':self.turb_rating_mw*1000,
                'hub_height': self.hubht,'rotor_diameter':self.rot_diam},
                'pv':{'system_capacity_kw':self.solar_init_size_mw*1000}
                }
        hybrid_plant=hpp.make_hybrid_plant(technologies,site_obj,self.scenario)
        init_wind_gen = hpp.get_wind_generation(hybrid_plant)
        init_solar_gen = hpp.get_solar_generation(hybrid_plant)
        # hybrid_plant.pv.capacity_factor/100
        
        self.renewable_plant_cost=self.init_renewables_costs(cost_year)
        # h2_per_mw_wind,h2_per_mw_solar=self.capac_2_h2_grad(init_wind_gen,init_solar_gen,wind_init_size_mw,self.solar_init_size_mw)
        # wind_capac,lcoh_wind,lcoh_wind_bat,bat_size_mw,bat_size_hrs,wind_h2,wind_bat_h2=\
        keys = ['Solar Size [MW]','Wind Size [MW]','Bat Size [MW]','Bat Hrs','LCOH','H2']

        wind_size,pv_size,all_lcoh,all_h2,bat_size_mw,bat_size_hrs=\
        self.wind_capac_2_lcoh(init_wind_gen,wind_init_size_mw,cost_year,init_solar_gen,self.solar_init_size_mw)
        w_vals = [pv_size,wind_size,bat_size_mw,bat_size_hrs,all_lcoh,all_h2]
        df=pd.DataFrame(dict(zip(keys,w_vals)))
        # wind_mult = wind_capac[np.argmin(lcoh_wind)]/wind_init_size_mw
        wind_mult = wind_size[np.argmin(all_lcoh)]/wind_init_size_mw
        

        # pv_capac,lcoh_pv,lcoh_pv_bat,bat_size_mw_with_pv,bat_size_hrs_with_pv,pv_h2,pv_bat_h2=\
        wind_size_pv,pv_size_pv,all_lcoh_pv,all_h2_pv,bat_size_mw_pv,bat_size_hrs_pv=\
        self.solar_capac_2_lcoh(init_solar_gen,self.solar_init_size_mw,cost_year,init_wind_gen*wind_mult,wind_size[np.argmin(all_lcoh)])
        
        # w_keys=['Solar Size [MW]','Wind Size [MW]','LCOH (wind)','H2 (wind)','LCOH (+ bat)','H2 (+bat)','Bat Size [MW]','Bat Hrs']
        # w_vals = [[self.solar_init_size_mw]*len(wind_capac),wind_capac,lcoh_wind,wind_h2,lcoh_wind_bat,wind_bat_h2,bat_size_mw,bat_size_hrs]

        # w_pv_keys=['Solar Size [MW]','Wind Size [MW]','LCOH (wind + pv)','H2 (wind + pv)','LCOH (+ bat)','H2 (+bat)','Bat Size [MW]','Bat Hrs']
        # w_pv_vals = [pv_capac,[wind_capac[np.argmin(lcoh_wind)]]*len(pv_capac),lcoh_pv,pv_h2,lcoh_pv_bat,pv_bat_h2,bat_size_mw_with_pv,bat_size_hrs_with_pv]
        # wpv_keys = ['Solar Size [MW]','Wind Size [MW]','Bat Size [MW]','Bat Hrs','LCOH','H2']
        wpv_vals = [pv_size_pv,wind_size_pv,bat_size_mw_pv,bat_size_hrs_pv,all_lcoh_pv,all_h2_pv]
        df=pd.concat([df,pd.DataFrame(dict(zip(keys,wpv_vals)))])
        df=df.reset_index(drop=True)
        return df
        # pd.DataFrame(dict(zip(wpv_keys,wpv_vals)))
        

        

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
        wind_specs = [self.hubht,self.turb_rating_mw,\
        wind_filename, self.rot_diam]
        wind_keys = ['Tower Height','Turbine Rating','Powercurve File','Rotor Diameter']
        vals = policy_vals + wind_specs
        keys = policy_keys + wind_keys
        scenario = dict(zip(keys,vals))
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










