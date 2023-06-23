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
from analysis import run_hybrid_plant
from scipy import interpolate
from tools.floris_2_csv import csv_from_floris
#from math import cos, asin, sqrt, pi
warnings.filterwarnings("ignore")
# import yaml

class national_sweep:
#Step 1: initialize inputs
#     a: costs
#     b: design
#     c: load profile (opt)
    def __init__(self,download_api,api_call_day):
        #50k points is 2km resolution
        #can use 10-50km resolution

        #west coast lon = -124.58
        #east coast lon = -67.924
        #northern lat: 49.54
        #southern lat: 25.62533
        

        self.parent_path = os.path.abspath('') + '/'
        self.main_dir = self.parent_path + 'NATIONAL_SWEEP/'
        wtk_file = self.main_dir + 'wtk_site_metadata.csv'
        self.resource_directory = self.parent_path + 'resource_files/'
        # DEFAULTS #
        default_input_file = self.main_dir + 'sweep_defaults.csv'
        self.model_params = pd.read_csv(default_input_file,index_col = 'Variable')
        self.resource_year = 2013
        self.annual_hydrogen_required_kg = 66000*1000 #metric ton-> kg

        #TODO: read in electrolyzer costs!
        #66,000 metric tons of hydrogen per year.
        #self.loc_file = pd.read_csv(wtk_file)
        #simple_sites=self.simple_get_sites() #use this to downsample sites
        #^requries self.loc_file to be read in!
        self.sample_site = pd.read_pickle(self.parent_path + 'hybrid/sites/sample_site').to_dict()
        self.hubht = self.model_params.loc['Hub Height'].values[0]
        # self.locs = pd.read_csv(self.main_dir + 'wtk_simple_sites_7kadj_onshore.csv')
        self.locs = pd.read_csv(self.main_dir + 'wtk_simple_sites_7kadj_onshore_enoughArea.csv')
        []
        # ua=self.locs['fraction_of_usable_area'].values
        # hm=[i for i,u in enumerate(ua) if u>0.1]
        #df.sort_values(by='State')['State']
        # idx_onshore = [i for i,state in enumerate(states) if state!='Unknown']
        
        #idx_good_lat=np.argwhere(self.locs['longitude'].values>-124.58)
        # self.quick_make_lbw_pysam_file()
        if download_api:
            df=self.stagger_api_calls(api_call_day)
            []
            
        []
    # def quick_make_lbw_pysam_file(self):
    #     csv_from_floris(self.main_dir + 'input_params/','lbw_6MW')
    #     []
    def approx_electrolyzer_capacity(self,wind_cf):
        hourly_h2_avg = self.annual_hydrogen_required_kg/8760
        bol_best_eff = [] #kWh/kg-H2
        eol_worst_eff = [] #kWh/kg-H2
        kWh_low = hourly_h2_avg*bol_best_eff #hourly energy needed
        kWh_high = hourly_h2_avg*eol_worst_eff #hourly energy needed
        installed_wind_capacity_kW = kWh_low
        # reqd_aep_low = kWh_low*8760
        # reqd_aep_high = kWh_high*8760
        # (reqd_aep_low/wind_cf)


        

    def stagger_api_calls(self,day_num):
        #states = self.locs['State'].values
        #lons=self.locs['longitude'].values
        states,state_cnt=np.unique(self.locs['State'].values,return_counts = True)
        statecnt_perday_api=np.floor(state_cnt/np.min(state_cnt))


        # lon_val,lon_cnt=np.unique(np.round(lons),return_counts=True)
        # locs_per_state = len(states)
        # num_calls=100
        # end_idx = start_idx + 1000
        # stagger=np.arange(start_idx,end_idx,num_calls)
        #df=pd.DataFrame()
        state_list = self.locs['State'].values
        # for start in stagger:
        #     df_stagger=self.get_resource_info(start,num_calls)
        site_idx_tracker=[]
        state_tracker=[]
        for sii,state in enumerate(states):
            all_state_idxes= [i for i,s in enumerate(state_list) if s==state]
            nsamples_for_this_state = int(statecnt_perday_api[sii])
            n_start = day_num*nsamples_for_this_state
            n_end = n_start + nsamples_for_this_state
            idx_to_use_today = all_state_idxes[n_start:n_end]
            state_info=self.locs.loc[idx_to_use_today]
            df_stagger=self.get_resource_info(state_info,state,day_num)
            state_tracker.extend(list(state_info['site_id'].values))
            site_idx_tracker.extend(list(state_info['State'].values))
            
            print("success - state {}".format(state))
            #df=pd.concat([df,df_stagger],axis=0)
        df=pd.DataFrame({'State':state_tracker,'site_id':site_idx_tracker})
        df.to_pickle(self.main_dir + 'ResourceDownloaded_info_day{}'.format(day_num))
        return df
    def get_resource_info(self,state_info,state,day_num):
        site_cnt=[]
        #indexes=np.arange(start_idx,start_idx+num_calls+1,1)
        sites=[]
        lats=[]
        lons=[]
        ids=[]
        indexes = list(state_info.index.values)
        for i in indexes:
            # lat=self.locs.iloc[i]['latitude']
            # lon=self.locs.iloc[i]['longitude']
            # id=self.locs.iloc[i]['site_id']
            lat=state_info.loc[i]['latitude']
            lon=state_info.loc[i]['longitude']
            id=state_info.loc[i]['site_id']
            site=self.make_site(lat,lon,self.sample_site)
            lats.append(lat)
            lons.append(lon)
            ids.append(id)
            sites.append(site)

        print("Last index was {} for site ID {}".format(i,id))
        df=pd.DataFrame({'Site ID':ids,'Lat':lats,'Lon':lons,'Site Obj':site})
        df.to_pickle(self.main_dir + 'Sites7k_{}_day{}'.format(state,day_num))

        return df
    # def get_resource_info(self,start_idx,num_calls):
    #     site_cnt=[]
    #     indexes=np.arange(start_idx,start_idx+num_calls+1,1)
    #     sites=[]
    #     lats=[]
    #     lons=[]
    #     ids=[]
    #     for i in indexes:
    #         lat=self.locs.iloc[i]['latitude']
    #         lon=self.locs.iloc[i]['longitude']
    #         id=self.locs.iloc[i]['site_id']
    #         site=self.make_site(lat,lon,self.sample_site)
    #         lats.append(lat)
    #         lons.append(lon)
    #         ids.append(id)
    #         sites.append(site)

    #     print("Last index was {} for site ID {}".format(i,id))
    #     df=pd.DataFrame({'Site ID':ids,'Lat':lats,'Lon':lons,'Site Obj':site})
    #     df.to_pickle(self.main_dir + 'Sites7k_num{}to{}'.format(start_idx,i))

    #     return df
    
    def make_site(self,lat,lon,sample_site):
        #print("[3]: making site...")
        
        sample_site['lat']=lat
        sample_site['lon']=lon
        sample_site['year']= self.resource_year
        sample_site['no_wind'] = False
        sample_site['no_solar'] = False
        #is below step necessary?
        
        site = SiteInfo(sample_site,resource_dir = self.resource_directory,hub_height=self.hubht)
        #self.main_dict['site']=site
        return site
    def simple_get_sites(self):
        lon_rnd=2
        lats = self.loc_file['latitude'].values
        latval,latcnt=np.unique(np.round(lats,3),return_counts=True) #21350 lats
        lons = self.loc_file['longitude'].values #44404 unique vals when round at 3
        lonval,loncnt=np.unique(np.round(lons,lon_rnd),return_counts=True)
        #cnts,cntcnt=np.unique(loncnt,return_counts=True)
        site_ids=[]
        for ilo, lon in enumerate(lonval):
            idx_lons=np.argwhere(np.round(lons,lon_rnd)==lon).reshape(loncnt[ilo])
            
            if loncnt[ilo]==1:
                idx=list(np.argwhere(np.round(lons,lon_rnd)==lon)[0])
                []
            elif loncnt[ilo]==2:
                #idxs=np.argwhere(np.round(lons,lon_rnd)==lon)
                # if np.isclose(lats[idxs[0]],lats[idxs[1]],atol=1):
                if np.isclose(lats[idx_lons[0]],lats[idx_lons[1]],atol=1):
                    idx=[idx_lons[0]]
                else:
                    idx=list(idx_lons)
                []
            else:
                
                lats_2check=np.round(lats[idx_lons])
                lat2_val,lat2cnt_idx=np.unique(lats_2check,return_index=True)
                
                newlats=lats[idx_lons][lat2cnt_idx].reshape(len(lat2cnt_idx))
                
                idx=[]
                for newlat in newlats:
                    lat_idx=np.argwhere(lats==newlat)
                    lat_idx=lat_idx.reshape(len(lat_idx))
                    idx_temp=[lati for lati in lat_idx if lati in idx_lons]
                    idx.extend(idx_temp)
                []
                
            dbl_cnt=[i for i in idx if i in site_ids]
            
            if len(dbl_cnt)>0:
                idx.remove(dbl_cnt)
            site_ids.extend(idx)
        cutdown_sites = self.loc_file.iloc[site_ids]
        return cutdown_sites



        #long has 5700 when round to 2
        #could do return_index = true
        #num_sites = len(latval)
    def get_sites_to_use(self,resolution_km):
        #https://stackoverflow.com/questions/27928/calculate-distance-between-two-latitude-longitude-points-haversine-formula
        p = np.pi/180
        #r = 6371 #earth radius [km]
        ed=12742 #2*earth radius [km]
        #US_area_kmsq=6629091 or (9.834*1e6)
        # a=(np.sin(resolution_km/ed))**2 #verified
        lat1=41.659
        lat2=41.659 #other opt is 46.9175
        lon1=-107.416
        lon2=-96.9487
        #lats are ==, d=869.0387,a=0.004644400187248377
        a = 0.5 - np.cos((lat2-lat1)*p)/2 + np.cos(lat1*p) * np.cos(lat2*p) * (1-np.cos((lon2-lon1)*p))/2
        d=ed * np.arcsin(np.sqrt(a)) 
        []

        return 
        

    def run_offgrid(self,input_df):
        use_simple_battery_dispatch=True #if true, ignore hydrogen demand for battery dispatch
        #placeholder to allow for battery dispatch independent of 
        #hydrogen demand. 

        #initialize renewable plant 1) sizes and 2) costs
        #TODO: have init_hybrid_plant make technologies variable used in run_hybrid_plant
        RE_info,renewable_cost_dict=self.init_hybrid_plant(input_df['Generation'])
        #pull out hub height
        hubht = RE_info['wind']['Hub Height']
        #initalize "scenario" - includes turbine info required for hybrid_plant
        scenario = self.init_scenario(input_df['site info'],RE_info['wind'])
        #create site object
        site=self.make_site(input_df['site info'],hubht)
       
        #cost_to_buy_kWH_from_grid shouldn't be used but doesn't hurt to have it
        cost_to_buy_kWh_from_grid=self.init_grid(input_df['Generation'])
        #grid and demand should be unused UNLESS there is intelligent dispatch control
        hydrogen_demand_profile =self.init_hydrogen_demand(input_df['site info'])
        #load_kWh_for_hydrogen shouldn't be used ?
        if use_simple_battery_dispatch:
            electrolyzer_capacity_kW=input_df['Hydrogen Production']['installed_electrolyzer_capacity']*1000
            min_load_kWh_for_hydrogen=0.1*electrolyzer_capacity_kW*np.ones(8760)
            max_load_kWh_for_hydrogen=electrolyzer_capacity_kW*np.ones(8760)
        #NOTE: change load_kWh_for_hydrogen
        load_kWh_for_hydrogen=self.init_pem(input_df['Hydrogen Production'],hydrogen_demand_profile)
        # if self.grid_connection_scenario =='off-grid':
        #     electrolyzer_capacity_kW=input_df['Hydrogen Production']['installed_electrolyzer_capacity']*1000
        #     min_load_kWh_for_hydrogen=0.1*electrolyzer_capacity_kW*np.ones(8760)
        #     max_load_kWh_for_hydrogen=electrolyzer_capacity_kW*np.ones(8760)
        #renewable_plant_performance_info has generation time-series
        #TODO: off-grid should have different 
        #NOTE: should off-grid have no hydrogen dispatch control strategy to meet demand?
        #aka - should hydrogen demand profile be used in off-grid case? If so, how strongly?
        renewable_plant_performance_info = self.run_wind_solar(RE_info,site,scenario,load_kWh_for_hydrogen)
        renewable_plant_generation=renewable_plant_performance_info['Wind + PV + Battery']
        
        H2_Results,h2_ts,h2_tot,energy_input_to_electrolyzer=self.run_electrolyzer(input_df['Hydrogen Production'],hydrogen_demand_profile,renewable_plant_generation)
        hydrogen_dispatched=self.run_hydrogen_dispatch(site,input_df['Hydrogen Storage'],H2_Results['hydrogen_hourly_production'],hydrogen_demand_profile)
        h2_solution,h2_summary,profast_h2_price_breakdown,lcoh_breakdown,extra_lcoh_info=self.run_lcoh_calc(input_df['Hydrogen Storage'],input_df['Hydrogen Transport'],cost_to_buy_kWh_from_grid,energy_input_to_electrolyzer,renewable_plant_generation,H2_Results,renewable_cost_dict)     


    def init_scenario(self,site_df,wind_info):
        print("[2]: initializing scenario...")
        #not used?
        if 'no' in site_df['policy option']:
            policy_vals = [0,0,0,0]
            policy_desc = 'no policy'
        elif 'max' in site_df['policy option']:
            policy_vals = [0,0.03072,3.0,0.5]
            policy_desc='max'
        else:
            policy_vals = [0,0,0,0]
        wind_specs = [wind_info['Hub Height'],wind_info['turbine_rating_mw'],\
        wind_info['filename'], wind_info['Rotor Diameter']]
        keys = ['Wind ITC','Wind PTC','H2 PTC','Storage ITC','Tower Height','Turbine Rating','Powercurve File','Rotor Diameter']
        vals = policy_vals + wind_specs
        scenario = dict(zip(keys,vals))
        self.main_dict['scenario']=scenario
        self.policy_option = policy_desc
        #keys = ['grid_connection_case','policy option','useful_life','debt_equity_split','discount_rate']
        return scenario
    
        
        
    
    

 
    

    def calc_desal_costs(self,electrolyzer_size_mw,kWh_per_kgH2):
        print("calculating desasl costs...")
        #Equations from desal_model
        #Values for CAPEX and OPEX given as $/(kg/s)
        #Source: https://www.nrel.gov/docs/fy16osti/66073.pdf
        #Assumed density of recovered water = 997 kg/m^3

        m3_water_per_kg_h2 = 0.01
        #desired fresh water flow rate [m^3/hr]
        desal_sys_size = electrolyzer_size_mw * (1000/kWh_per_kgH2) * m3_water_per_kg_h2 #m^3

        desal_opex = 4841 * (997 * desal_sys_size / 3600) # Output in USD/yr
        desal_capex = 32894 * (997 * desal_sys_size / 3600) # Output in USD

        return desal_capex,desal_opex
        

    
#from elenya_write_outputs import esg_write_outputs_ProFAST
    def write_outputs(self):
        print("writing outputs...")
        pass



if __name__ == "__main__":
    start_indx = 0
    api_call_day=0
    sweep = national_sweep(download_api=True,api_call_day=api_call_day)










