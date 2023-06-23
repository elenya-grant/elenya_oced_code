# from analysis.hopp_for_h2 import hopp_for_h2
from analysis.simple_dispatch import SimpleDispatch
# from analysis.hopp_for_h2_floris import hopp_for_h2_floris
from analysis.run_h2_PEM import run_h2_PEM
import numpy as np
import yaml
class electrolyzer_tools:
    def __init__(self,grid_connection_scenario,EOL_eff_drop,electrolyzer_CapEx,plant_life):
        #defaults not intended for optimization for PEM
        self.grid_connection_scenario = grid_connection_scenario
        self.pem_control_type ='basic'
        self.use_degradation_penalty=True
        self.plant_life=plant_life
        self.electrolyzer_direct_cost_kw=electrolyzer_CapEx
        self.electrolysis_scale='Distributed' #unused in run_h2_PEM
        #self.user_defined_pem_param_dictionary=user_defined_params

        self.user_defined_pem_param_dictionary={ 
                'Modify BOL Eff':False,
                'BOL Eff [kWh/kg-H2]':[],
                'Modify EOL Degradation Value':True,
                'EOL Rated Efficiency Drop':EOL_eff_drop}

        #added

    def run_electrolyzer(self,electrolyzer_size_mw,n_pem_clusters,hydrogen_production_capacity_required_kgphr,electrical_generation_timeseries):
        #this works for 1) off-grid and 2) constant hydrogen demand
        H2_Results, h2_ts, h2_tot,energy_input_to_electrolyzer=run_h2_PEM(electrical_generation_timeseries, electrolyzer_size_mw,\
            self.plant_life, n_pem_clusters,  self.electrolysis_scale, \
            self.pem_control_type,self.electrolyzer_direct_cost_kw, self.user_defined_pem_param_dictionary,\
            self.use_degradation_penalty, self.grid_connection_scenario,\
            hydrogen_production_capacity_required_kgphr)
    
        return H2_Results,h2_ts,h2_tot,energy_input_to_electrolyzer
    def init_opt_electrolyzer(self,electrolyzer_size_mw,n_pem_clusters,annual_h2_required):
        from analysis.run_PEM_master import run_PEM_clusters
        fake_power = np.zeros(8760)
        self.pem_master = run_PEM_clusters(fake_power,electrolyzer_size_mw,n_pem_clusters,self.electrolyzer_direct_cost_kw,self.plant_life,self.user_defined_pem_param_dictionary,self.use_degradation_penalty)
        self.annual_h2_required = annual_h2_required
        
    # def run_opt_electrolyzer(self,electrolyzer_size_mw,n_pem_clusters,hydrogen_production_capacity_required_kgphr,electrical_generation_timeseries):
        #this works for 1) off-grid and 2) constant hydrogen demand
        # H2_Results, h2_ts, h2_tot,energy_input_to_electrolyzer=run_h2_PEM(electrical_generation_timeseries, electrolyzer_size_mw,\
        #     self.plant_life, n_pem_clusters,  self.electrolysis_scale, \
        #     self.pem_control_type,self.electrolyzer_direct_cost_kw, self.user_defined_pem_param_dictionary,\
        #     self.use_degradation_penalty, self.grid_connection_scenario,\
        #     hydrogen_production_capacity_required_kgphr)
    def run_simple_opt_electrolyzer(self,electrical_generation_timeseries):
        #H2_results = self.make_h2_results(h2_df_ts,h2_df_tot)
        self.pem_master.input_power_kw = electrical_generation_timeseries
        h2_df_ts,h2_df_tot=self.pem_master.run()
        H2_results = self.make_h2_results(h2_df_ts,h2_df_tot)
        annual_h2 = np.sum(h2_df_ts.loc['hydrogen_hourly_production'].sum())
        #output_info = {'H2_Results':H2_results,'h2_tot':h2_df_tot
        # return H2_results,h2_df_tot,annual_h2
        return H2_results,annual_h2
    def run_full_opt_electrolyzer(self,electrical_generation_timeseries):
        #H2_results = self.make_h2_results(h2_df_ts,h2_df_tot)
        self.pem_master.input_power_kw = electrical_generation_timeseries
        h2_df_ts,h2_df_tot=self.pem_master.run()
        H2_results = self.make_h2_results(h2_df_ts,h2_df_tot)
        # annual_h2 = np.sum(h2_df_ts.loc['hydrogen_hourly_production'].sum())
        #output_info = {'H2_Results':H2_results,'h2_tot':h2_df_tot
        # return H2_results,h2_df_tot,annual_h2
        return H2_results,h2_df_tot
    def run_opt_electrolyzer(self,electrical_generation_timeseries):

        self.pem_master.input_power_kw = electrical_generation_timeseries
        h2_df_ts,h2_df_tot=self.pem_master.run()
        hydrogen_hourly_production = h2_df_ts.loc['hydrogen_hourly_production'].sum()
        hydrogen_annual_production = np.sum(hydrogen_hourly_production)
        # avg_daily_h2 = np.sum(hydrogen_hourly_production)/8760
        if np.isclose(hydrogen_annual_production,self.annual_h2_required,rtol=0.05):
            H2_results = self.make_h2_results(h2_df_ts,h2_df_tot)
            success = True
            output_info = {'H2_Results':H2_results,'h2_ts':h2_df_ts,'h2_tot':h2_df_tot}
        else:
            extra_h2,extra_power,idx_h2_made = self.check_annual_h2(hydrogen_hourly_production,electrical_generation_timeseries)
            output_info = {'H2 Error':extra_h2,'Power Error':extra_power,'Idx':idx_h2_made,'Annual H2':hydrogen_annual_production}
            success = False
        return success,output_info#H2_Results,h2_ts,h2_tot
    def check_annual_h2(self,hydrogen_hourly_production,electrical_generation_timeseries):
        extra_h2 =np.sum(hydrogen_hourly_production)-self.annual_h2_required
        # if np.sum(hydrogen_hourly_production)>annual_h2_required:
        if extra_h2>0:
            #made too much hydrogen
            h2_cumulative = np.cumsum(hydrogen_hourly_production)
            idx_made_enough_h2 = np.argwhere(h2_cumulative>self.annual_h2_required)[:,0][0]
            extra_power = np.sum(electrical_generation_timeseries[idx_made_enough_h2:len(electrical_generation_timeseries)])
            
        else:
            #didn't make enough hydrogen
            #xtra_h2_needed
            # extra_h2 =extra_h2
            #annual_h2_required-np.sum(hydrogen_hourly_production)
            avg_eff = np.sum(electrical_generation_timeseries)/np.sum(hydrogen_hourly_production)
            #extra power needed
            extra_power = -1*extra_h2*avg_eff
            idx_made_enough_h2=None
        return extra_h2,extra_power,idx_made_enough_h2


    def make_h2_results(self,h2_ts,h2_tot):
        energy_input_to_electrolyzer=h2_ts.loc['Input Power [kWh]'].sum()
        average_uptime_hr=h2_tot.loc['Total Uptime [sec]'].mean()/3600
        elec_rated_h2_capacity_kgpy =h2_tot.loc['Cluster Rated H2 Production [kg/yr]'].sum()
        cap_factor=h2_tot.loc['Total H2 Production [kg]'].sum()/elec_rated_h2_capacity_kgpy
        hydrogen_hourly_production = h2_ts.loc['hydrogen_hourly_production'].sum()
        water_hourly_usage = h2_ts.loc['water_hourly_usage_kg'].sum()
        water_annual_usage = np.sum(water_hourly_usage)
        hourly_system_electrical_usage=h2_ts.loc['Power Consumed [kWh]'].sum()
        total_system_electrical_usage = np.sum(hourly_system_electrical_usage)
        avg_eff_perc=39.41*hydrogen_hourly_production/hourly_system_electrical_usage #np.nan_to_num(h2_ts.loc['electrolyzer_total_efficiency_perc'].mean())
        hourly_efficiency=np.nan_to_num(avg_eff_perc)
        tot_avg_eff=39.41/h2_tot.loc['Total kWh/kg'].mean()
        hydrogen_annual_output = sum(hydrogen_hourly_production)

        rated_kWh_pr_kg=h2_tot.loc['Stack Rated Power Consumed [kWh]'].values[0]/h2_tot.loc['Stack Rated H2 Production [kg/hr]'].values[0]
        H2_Results = {'hydrogen_annual_output':
                     hydrogen_annual_output,
                  'cap_factor':
                     cap_factor,
                  'hydrogen_hourly_production':
                     hydrogen_hourly_production,
                  'water_hourly_usage':
                  water_hourly_usage,
                  'water_annual_usage':
                  water_annual_usage,
                  'electrolyzer_avg_efficiency':
                  tot_avg_eff,
                  'total_electrical_consumption':
                  total_system_electrical_usage,
                  'electrolyzer_total_efficiency':
                  hourly_efficiency,
                  'time_between_replacement_per_stack':
                  h2_tot.loc['Avg [hrs] until Replacement Per Stack'],
                  'avg_time_between_replacement':
                  h2_tot.loc['Avg [hrs] until Replacement Per Stack'].mean(),
                  'Rated kWh/kg-H2':rated_kWh_pr_kg,
                  'average_operational_time [hrs]':
                  average_uptime_hr
                  }
        return H2_Results

    def calc_energy_required_for_hourly_hydrogen_demand(self):
        pass

    def apply_extra_hydrogen_losses(self,H2_Results,h2_ts,h2_tot):
        pass
    def multi_battery_pem_basic_control(self):
        pass
    def smart_grid_dispatch(self):
        pass

    def simple_multibattery_dispatch(self,ramp_rate,capacity,n_bat):
        
        #to run NewSimpleDispatch
        # unit_dispatched, excess_units, storage_SOC=obj.run(shortfall,curtailment)
        pass
    def create_multidispatch_objects(self,ramp_rate,capacity,n_obj):
        from analysis.new_simple_dispatch import NewSimpleDispatch
        objs=[]
        if isinstance(n_obj,float):
            n_obj=int(n_obj)
        if isinstance(ramp_rate,(float,int)):
            ramp_rate = ramp_rate*np.ones(n_obj)
        if isinstance(capacity,(float,int)):
            capacity = capacity*np.ones(n_obj)
        for i in range(n_obj):
            single_obj = NewSimpleDispatch(ramp_rate[i],capacity[i])
            objs.append(single_obj)
        return objs

