# from hybrid.PEM_H2_LT_electrolyzer import PEM_electrolyzer_LT
import numpy as np
import pandas as pd
from analysis.run_PEM_master import run_PEM_clusters
def clean_up_final_outputs(h2_tot,h2_ts):
   h2_tot.loc["Life"]
   new_h2_tot = h2_tot.drop(['Cluster Rated H2 Production [kg/hr]','Cluster Rated Power Consumed [kWh]','Cluster Rated H2 Production [kg/yr]',\
      'Stack Rated H2 Production [kg/hr]','Stack Rated Power Consumed [kWh]',"Stack Life Summary","Life"])
   h2_ts.sum(axis=1)
   ts_sum_desc = ['Input Power [kWh]','Power Consumed [kWh]',\
      'hydrogen production no start-up time','hydrogen_hourly_production',\
         'water_hourly_usage_gal','water_hourly_usage_kg','Stacks on']
   
   new_h2_ts = h2_ts.drop(['V_cell With Deg','Power Per Stack [kW]','Stack Current [A]'])
   new_h2_ts = new_h2_ts.loc[ts_sum_desc].sum(axis=1)
   
   # new_h2_ts.loc[ts_avg_desc].mean(axis=1)
   # pd.concat([new_h2_ts.loc[ts_sum_desc].sum(axis=1),new_h2_ts.loc[ts_avg_desc].mean(axis=1)])
   return new_h2_ts,new_h2_tot
def clean_up_lifetime_performance(h2_tot,use_degradation_penalty):
   cluster_names = list(h2_tot.columns.values)
   t_eod_metrics = list(h2_tot.loc["Life"]["Cluster #0"].index.values)
   if use_degradation_penalty:
      lifetime_desc = 'full losses'
   else:
      lifetime_desc = 'warm-up losses'

   # stack_life_df = pd.DataFrame()
   redo_performance_df = pd.DataFrame()
   for ci,cname in enumerate(cluster_names):
      
      # h2_tot.loc["Stack Life Summary"].loc[cname].name = cname
      # stack_life_df = pd.concat([stack_life_df,h2_tot.loc["Stack Life Summary"].loc[cname]],axis = 1)

      for t_eod_type in t_eod_metrics:
         temp_df = pd.DataFrame(h2_tot.loc["Life"][cname].loc[t_eod_type].loc[lifetime_desc]).T
         temp_df.index = [[cname],[t_eod_type]]
         redo_performance_df = pd.concat([redo_performance_df,temp_df],axis=0)
   
   cluster_average_keys = ['Capacity Factor [-]','Average Efficiency [kWh/kg]','Stack Life [hrs]']
   cluster_sum_keys = ['Total Hydrogen Produced [kg]','Average Annual Hydrogen Produced [kg]']
   cluster_avg_df = pd.DataFrame()
   for t_eod_type in t_eod_metrics:
      redo_performance_df.swaplevel().loc[t_eod_type]
      redo_performance_df.swaplevel().loc[t_eod_type][cluster_average_keys].mean()
      redo_performance_df.swaplevel().loc[t_eod_type][cluster_sum_keys].sum()
      ddf=pd.concat([redo_performance_df.swaplevel().loc[t_eod_type][cluster_average_keys].mean(),redo_performance_df.swaplevel().loc[t_eod_type][cluster_sum_keys].sum()],axis=0)
      ddf.name = t_eod_type
      cluster_avg_df = pd.concat([cluster_avg_df,ddf],axis=1)
   
   # return stack_life_df,cluster_avg_df.drop('Total Hydrogen Produced [kg]'),redo_performance_df
   return cluster_avg_df.drop('Total Hydrogen Produced [kg]'),redo_performance_df

def run_h2_PEM(electrical_generation_timeseries, electrolyzer_size,
                useful_life, n_pem_clusters,  electrolysis_scale, 
                pem_control_type,electrolyzer_direct_cost_kw, user_defined_pem_param_dictionary,
                use_degradation_penalty, grid_connection_scenario,
                hydrogen_production_capacity_required_kgphr
                ):
   #base stack life estimates on operational time ("Active") vs simulation length ("Sim")
   life_spec_options = ["Active","Sim"]
   #base stack life estimates on rated efficiency loss ("Optimistic") 
   #or distribution of operational load ("Conservative")
   operation_spec_options = ["Optimistic","Conservative"]
   #TODO: make the below things inputs, but setting as defaults right now
   debug_mode = False #if False, then the outputs are cleaned  up
   life_spec = "Sim" #more conservative than "Active"
   op_spec = "Optimistic" #based on rated, more optimistic than "Conservative"
   
   stack_life_spec_desc = '{} - {}'.format(op_spec,life_spec)
   
   #initialize electrolyzers
   pem=run_PEM_clusters(electrical_generation_timeseries,electrolyzer_size,n_pem_clusters,electrolyzer_direct_cost_kw,useful_life,user_defined_pem_param_dictionary,use_degradation_penalty)

   if grid_connection_scenario!='off-grid':
      h2_ts,h2_tot=pem.run_grid_connected_pem(electrolyzer_size,hydrogen_production_capacity_required_kgphr)
   else:
      if pem_control_type == 'optimize':
         h2_ts,h2_tot=pem.run(optimize=True)
      else:
         #run basic controller
         h2_ts,h2_tot=pem.run()
   #avg_pem_cf = np.mean(h2_tot.loc['PEM Capacity Factor'].values)

   #reorganize the outputs
   # stack_life_df,cluster_avg_df,reformatted_life_df = clean_up_lifetime_performance(h2_tot,use_degradation_penalty)
   cluster_avg_df,reformatted_life_df = clean_up_lifetime_performance(h2_tot,use_degradation_penalty)
   
   # energy_used_by_electrolyzer=h2_ts.loc['Power Consumed [kWh]'].sum()
   energy_input_to_electrolyzer=h2_ts.loc['Input Power [kWh]'].sum()
   average_uptime_hr=h2_tot.loc['Total Uptime [sec]'].mean()/3600
   # avg_generation = np.mean(electrical_generation_timeseries)  # Avg Generation
   
   elec_rated_h2_capacity_kgpy =h2_tot.loc['Cluster Rated H2 Production [kg/yr]'].sum() 
   
   #cap_factor = h2_tot.loc['PEM Capacity Factor'].mean()#avg_generation / kw_continuous
   # This appears to give the same answer but it is a better definition
   cap_factor=h2_tot.loc['Total H2 Production [kg]'].sum()/elec_rated_h2_capacity_kgpy

   hydrogen_hourly_production = h2_ts.loc['hydrogen_hourly_production'].sum()
   water_hourly_usage = h2_ts.loc['water_hourly_usage_kg'].sum()
   water_annual_usage = np.sum(water_hourly_usage)
   hourly_system_electrical_usage=h2_ts.loc['Power Consumed [kWh]'].sum()
   total_system_electrical_usage = np.sum(hourly_system_electrical_usage)
   avg_eff_perc=39.41*hydrogen_hourly_production/hourly_system_electrical_usage #np.nan_to_num(h2_ts.loc['electrolyzer_total_efficiency_perc'].mean())
   hourly_efficiency=np.nan_to_num(avg_eff_perc)
   tot_avg_eff=39.41/h2_tot.loc['Total kWh/kg'].mean()
   
   max_h2_pr_hr = h2_tot.loc['Cluster Rated H2 Production [kg/hr]'].sum()
   max_pwr_pr_hr = h2_tot.loc['Cluster Rated Power Consumed [kWh]'].sum()
   rated_kWh_pr_kg = h2_tot.loc['Stack Rated Efficiency [kWh/kg]'].mean()
   elec_rated_h2_capacity_kgpy =h2_tot.loc['Cluster Rated H2 Production [kg/yr]'].sum()
   
   atrribute_desc = ["Efficiency [kWh/kg]","H2 Production [kg/hr]","Power Consumed [kWh]","Annual H2 Production [kg/year]"]
   sim = ["Capacity Factor","Active Time / Sim Time","Total Input Power [kWh]",\
      "Total H2 Produced [kg]","Average Efficiency [kWh/kg]",\
      "Average Efficiency [%-HHV]","Total Stack Off-Cycles","H2 Warm-Up Losses [kg]"]
   
   sim_specs = ['Sim: '+s for s in sim]
   attribute_specs = ['Rated BOL: '+s for s in atrribute_desc]
   life_spec_desc = ['Life: ' + s for s in list(cluster_avg_df.index.values)]
   # life = cluster_avg_df[stack_life_spec_desc].to_dict()
   attributes = [rated_kWh_pr_kg,max_h2_pr_hr,max_pwr_pr_hr,elec_rated_h2_capacity_kgpy]
   sim_performance = [h2_tot.loc['Total kWh/kg'].mean(),h2_tot.loc['Operational Time / Simulation Time (ratio)'].mean(),h2_tot.loc['Total Input Power [kWh]'].sum(),\
      h2_tot.loc['Total H2 Production [kg]'].sum(),h2_tot.loc['Total kWh/kg'].mean(),\
      tot_avg_eff,h2_tot.loc['Total Off-Cycles'].sum(),h2_tot.loc['Warm-Up Losses on H2 Production'].sum()]
   new_H2_Results = dict(zip(attribute_specs,attributes))
   new_H2_Results.update(dict(zip(sim_specs,sim_performance)))
   new_H2_Results.update(dict(zip(life_spec_desc,cluster_avg_df[stack_life_spec_desc].values)))
   new_H2_Results['Stack Life - All Options'] = reformatted_life_df
   
   #below is so that downstream code doesn't break
   system_avg_annual_h2_kg_pr_year = cluster_avg_df[stack_life_spec_desc]['Average Annual Hydrogen Produced [kg]']
   system_avg_life_capfac = cluster_avg_df[stack_life_spec_desc]['Capacity Factor [-]']
   system_avg_life_eff_perc= cluster_avg_df[stack_life_spec_desc]['Average Efficiency [kWh/kg]']
   average_stack_life_hrs = cluster_avg_df[stack_life_spec_desc]['Stack Life [hrs]']
   H2_Results = {'max_hydrogen_production [kg/hr]':
                  max_h2_pr_hr,
                  'hydrogen_annual_output':
                     system_avg_annual_h2_kg_pr_year,
                  'cap_factor':
                  system_avg_life_capfac,
                  'cap_factor_sim':
                     cap_factor,
                  'hydrogen_hourly_production':
                     hydrogen_hourly_production,
                  'water_hourly_usage':
                  water_hourly_usage,
                  'water_annual_usage':
                  water_annual_usage,
                  'electrolyzer_avg_efficiency':
                  system_avg_life_eff_perc,
                  # tot_avg_eff,
                  'total_electrical_consumption':
                  total_system_electrical_usage,
                  'electrolyzer_total_efficiency':
                  hourly_efficiency,
                  # 'time_between_replacement_per_stack':
                  # h2_tot.loc['Avg [hrs] until Replacement Per Stack'],
                  'avg_time_between_replacement':
                  average_stack_life_hrs,
                  # h2_tot.loc['Avg [hrs] until Replacement Per Stack'].mean(),
                  'Rated kWh/kg-H2':rated_kWh_pr_kg,
                  'average_operational_time [hrs]':
                  average_uptime_hr,
                  'new_H2_Results':new_H2_Results
                  }

   
   if not debug_mode:
      h2_ts,h2_tot = clean_up_final_outputs(h2_tot,h2_ts)
  
   return H2_Results, h2_ts, h2_tot,energy_input_to_electrolyzer