# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 12:31:28 2022

@author: ereznic2
"""
# Specify file path to PyFAST
import sys
#sys.path.insert(1,'../PyFAST/')
import numpy as np
import pandas as pd
# sys.path.insert(1,sys.path[0] + '/ProFAST-main/') #ESG
sys.path.insert(1,'/Users/egrant/Desktop/HOPP-GIT/HOPP/ProFAST-main/') #ESG
import ProFAST

#from analysis.LCA_single_scenario_ProFAST import hydrogen_LCA_singlescenario_ProFAST

#sys.path.append('../ProFAST/')

pf = ProFAST.ProFAST()


def opt_run_profast_for_hydrogen(total_direct_electrolyzer_cost_kw,\
                            electrolyzer_size_mw,H2_Results,\
                            hydrogen_storage_capacity_kg,hydrogen_storage_cost_USDprkg,\
                            renewable_plant_cost_info,energy_from_renewables_kWh,\
                            policy_option,plant_life,water_cost, \
                            wind_size_mw,solar_size_mw,battery_size_mw,battery_hrs):

    # Electrolyzer Defaults
    fixed_OM = 12.8 #[$/kW-y]
    total_variable_OM = 1.3#[$/MWh]
    stack_replacement_cost = 15/100  #[% of installed capital cost]
    # electrolyzer_installation_factor = 12/100  #[%] for stack cost 

    #workaround desal costs
    #desal_sys_size=electrolyzer_size_mw * (10/55.5)
    opex_desal = 1340.6880555555556*electrolyzer_size_mw * (10/55.5)
    capex_desal =9109.810555555556*electrolyzer_size_mw * (10/55.5)

    

    
    electrolyzer_efficiency_while_running = []
    # water_consumption_while_running = []
    hydrogen_production_while_running = []
    for j in range(len(H2_Results['electrolyzer_total_efficiency'])):
        if H2_Results['hydrogen_hourly_production'][j] > 0:
            electrolyzer_efficiency_while_running.append(H2_Results['electrolyzer_total_efficiency'][j])
            # water_consumption_while_running.append(H2_Results['water_hourly_usage'][j])
            hydrogen_production_while_running.append(H2_Results['hydrogen_hourly_production'][j])
    
    #electrolyzer_design_efficiency_HHV = np.max(electrolyzer_efficiency_while_running) # Should ideally be user input
    electrolyzer_average_efficiency_HHV = np.mean(electrolyzer_efficiency_while_running)
    
    # water_consumption_avg_galH2O_prkgH2 = water_consumption_avg_kgH2O_prkgH2/3.79
    water_consumption_avg_galH2O_prkgH2=10/3.79
    # Calculate average electricity consumption from average efficiency
    h2_HHV = 141.88
    elec_avg_consumption_kWhprkg = h2_HHV*1000/3600/electrolyzer_average_efficiency_HHV
    
    # Design point electricity consumption
    elec_consumption_kWhprkg_design=H2_Results['Rated kWh/kg-H2']

    # Calculate electrolyzer production capacity
    #TODO: make below an input/calculation
    electrolysis_plant_capacity_kgperday=   electrolyzer_size_mw/elec_consumption_kWhprkg_design*1000*24
    #electrolysis_plant_capacity_kgperday = electrolyzer_size_mw*electrolyzer_design_efficiency_HHV/h2_HHV*3600*24
    
    # Installed capital cost
    
    
    # Indirect capital cost as a percentage of installed capital cost
    site_prep = 2/100   #[%]
    engineering_design = 10/100 #[%]
    project_contingency = 15/100 #[%]
    permitting = 15/100     #[%]
    land_cost = 250000   #[$]
    
    
    # Calculate electrolyzer installation cost
    #NOTE: this is the input given by OCED
    #total_direct_electrolyzer_cost_kw = (electrolyzer_system_capex_kw * (1+electrolyzer_installation_factor)) \
    
    electrolyzer_total_installed_capex = total_direct_electrolyzer_cost_kw*electrolyzer_size_mw*1000
    
    electrolyzer_indirect_cost = electrolyzer_total_installed_capex*(site_prep+engineering_design+project_contingency+permitting)
                                   
    
    # Calculate capital costs
    capex_electrolyzer_overnight = electrolyzer_total_installed_capex + electrolyzer_indirect_cost
    
    #hydrogen storage CapEx
    capex_storage_installed = hydrogen_storage_capacity_kg*hydrogen_storage_cost_USDprkg

    #hydrogen storage compressor costs
    compressor_capex_USDprkWe_of_electrolysis = 39
    capex_compressor_installed = compressor_capex_USDprkWe_of_electrolysis*electrolyzer_size_mw*1000


    fixed_cost_electrolysis_total = fixed_OM*electrolyzer_size_mw*1000
    property_tax_insurance = 1.5/100    #[% of Cap/y]
    
    total_variable_OM_perkg = total_variable_OM*elec_avg_consumption_kWhprkg/1000

    # electrolzyer stack replacement schedule
    
    refturb_period = np.max([1,int(np.floor(H2_Results['avg_time_between_replacement']/(24*365)))])

    electrolyzer_refurbishment_schedule = np.zeros(plant_life)
    electrolyzer_refurbishment_schedule[refturb_period:plant_life:refturb_period]=stack_replacement_cost
    

    #NOTE: add below in if using policy!
    # electrolysis_total_EI_policy_grid=0
    # electrolysis_total_EI_policy_offgrid=0
    # electrolysis_total_EI_policy_grid,electrolysis_total_EI_policy_offgrid\
    #       = hydrogen_LCA_singlescenario_ProFAST(cambiumdata_filepath,grid_connection_scenario,H2_Results['hydrogen_annual_output'],energy_from_grid_kWh,plant_life)
    
    grid_electricity_useage_kWhpkg = 0 #np.sum(energy_from_grid_kWh)/H2_Results['hydrogen_annual_output']
    ren_electricity_useage_kWhpkg =np.sum(energy_from_renewables_kWh)/H2_Results['hydrogen_annual_output']
    # ren_frac = 1#np.min([1,np.sum(energy_from_renewables_kWh)/np.sum(energy_to_electrolyzer)])
    
    elec_cf = H2_Results['cap_factor']
        
    
    if policy_option == 'no policy':
        Ren_PTC = 0 
    elif policy_option == 'max':
        Ren_PTC = 0.03072 * ren_electricity_useage_kWhpkg
    
    
    #wind_size_mw=renewable_plant_cost_info['wind']['size_mw']
    wind_om_cost_kw =  renewable_plant_cost_info['wind']['o&m_per_kw']
    fixed_cost_wind = wind_om_cost_kw*wind_size_mw*1000 
    capex_wind_installed = renewable_plant_cost_info['wind']['capex_per_kw'] * wind_size_mw*1000
    #wind_cost_adj = [val for val in renewable_plant_cost_info['wind_savings_dollars'].values()]
    #wind_revised_cost=0#np.sum(wind_cost_adj)

    #solar_size_mw=renewable_plant_cost_info['pv']['size_mw']
    solar_om_cost_kw = renewable_plant_cost_info['pv']['o&m_per_kw']
    fixed_cost_solar = solar_om_cost_kw*solar_size_mw*1000 
    capex_solar_installed = renewable_plant_cost_info['pv']['capex_per_kw'] * solar_size_mw*1000
    #battery_hrs=renewable_plant_cost_info['battery']['storage_hours']
    # battery_size_mw=['battery']['size_mw']
    battery_capex_per_kw= renewable_plant_cost_info['battery']['capex_per_kwh']*battery_hrs +  renewable_plant_cost_info['battery']['capex_per_kw']
    capex_battery_installed = battery_capex_per_kw * battery_size_mw*1000
    fixed_cost_battery = renewable_plant_cost_info['battery']['o&m_percent'] * capex_battery_installed

    

    #capex_wind_installed=capex_wind_installed_init-wind_revised_cost
    
    
    H2_PTC_duration = 10 # years the tax credit is active
    Ren_PTC_duration = 10 # years the tax credit is active
    
    # if policy_option == ('no-policy' or 'no policy'):
    if policy_option ==  'no policy':
        ITC = 0
        H2_PTC = 0 # $/kg H2
        Ren_PTC = 0 # $/kWh
        
    elif policy_option == 'max':
        
        ITC = 0.5
        H2_PTC = 0 #for OPED analysis
        
        # if electrolysis_total_EI_policy <= 0.45: # kg CO2e/kg H2
        #     H2_PTC = 3 # $/kg H2
        # elif electrolysis_total_EI_policy > 0.45 and electrolysis_total_EI_policy <= 1.5: # kg CO2e/kg H2
        #     H2_PTC = 1 # $/kg H2
        # elif electrolysis_total_EI_policy > 1.5 and electrolysis_total_EI_policy <= 2.5: # kg CO2e/kg H2     
        #     H2_PTC = 0.75 # $/kg H2
        # elif electrolysis_total_EI_policy > 2.5 and electrolysis_total_EI_policy <= 4: # kg CO2e/kg H2    
        #     H2_PTC = 0.6 # $/kg H2 
        # elif electrolysis_total_EI_policy > 4:
        #     H2_PTC = 0
                                
    #TODO: make grid_price_per_kWh an input!
    #grid_price_perkWh = mwh_to_kwh*elec_price # convert $/MWh to $/kWh     
    # Set up ProFAST
    grid_price_perkWh=0
    pf = ProFAST.ProFAST('blank')
    
    # Fill these in - can have most of them as 0 also
    gen_inflation = 0.00
    pf.set_params('commodity',{"name":'Hydrogen',"unit":"kg","initial price":100,"escalation":gen_inflation})
    pf.set_params('capacity',electrolysis_plant_capacity_kgperday) #units/day
    pf.set_params('maintenance',{"value":0,"escalation":gen_inflation})
    pf.set_params('analysis start year',2022)
    pf.set_params('operating life',plant_life)
    pf.set_params('installation months',36)
    pf.set_params('installation cost',{"value":0,"depr type":"Straight line","depr period":4,"depreciable":False})
    pf.set_params('non depr assets',land_cost)
    pf.set_params('end of proj sale non depr assets',land_cost*(1+gen_inflation)**plant_life)
    pf.set_params('demand rampup',0)
    pf.set_params('long term utilization',elec_cf)
    pf.set_params('credit card fees',0)
    pf.set_params('sales tax',0) 
    pf.set_params('license and permit',{'value':00,'escalation':gen_inflation})
    pf.set_params('rent',{'value':0,'escalation':gen_inflation})
    pf.set_params('property tax and insurance percent',property_tax_insurance)
    pf.set_params('admin expense percent',0)
    pf.set_params('total income tax rate',0.27)
    pf.set_params('capital gains tax rate',0.15)
    pf.set_params('sell undepreciated cap',True)
    pf.set_params('tax losses monetized',True)
    pf.set_params('operating incentives taxable',True)
    pf.set_params('general inflation rate',gen_inflation)
    pf.set_params('leverage after tax nominal discount rate',0.0824)
    pf.set_params('debt equity ratio of initial financing',1.38)
    pf.set_params('debt type','Revolving debt')
    pf.set_params('debt interest rate',0.0489)
    pf.set_params('cash onhand percent',1)
    pf.set_params('one time cap inct',{'value':np.nan_to_num(ITC*capex_storage_installed),'depr type':'MACRS','depr period':7,'depreciable':True})
    pf.set_params('one time cap inct',{'value':np.nan_to_num(ITC*capex_solar_installed),'depr type':'MACRS','depr period':7,'depreciable':True})
    pf.set_params('one time cap inct',{'value':np.nan_to_num(ITC*capex_battery_installed),'depr type':'MACRS','depr period':7,'depreciable':True})
    
    #----------------------------------- Add capital items to ProFAST ----------------
    #pf.add_capital_item(name="Electrolysis system",cost=capex_electrolyzer_overnight,depr_type="MACRS",depr_period=5,refurb=[0])
    pf.add_capital_item(name="Electrolysis system",cost=capex_electrolyzer_overnight,depr_type="MACRS",depr_period=7,refurb=list(electrolyzer_refurbishment_schedule))
    pf.add_capital_item(name="Compression",cost=capex_compressor_installed,depr_type="MACRS",depr_period=7,refurb=[0])
    pf.add_capital_item(name="Hydrogen Storage",cost=capex_storage_installed,depr_type="MACRS",depr_period=7,refurb=[0])
    pf.add_capital_item(name ="Desalination",cost = capex_desal,depr_type="MACRS",depr_period=7,refurb=[0])
    #ESG added below
    #pf.add_capital_item(name="Hydrogen Transport",cost=h2_transport_CapEx,depr_type="MACRS",depr_period=7,refurb=[0]) #added

    pf.add_capital_item(name = "Wind Plant",cost = capex_wind_installed,depr_type = "MACRS",depr_period = 7,refurb = [0])
    pf.add_capital_item(name = "Solar Plant",cost = capex_solar_installed,depr_type = "MACRS",depr_period = 7,refurb = [0])
    pf.add_capital_item(name = "Battery Storage",cost = capex_battery_installed,depr_type = "MACRS",depr_period = 7,refurb = [0])
    
    #NOTE TOTAL CAPEX DOES NOT REFLECT STACK REPLACEMENT COSTS AND HOW THEY CHANGE OVER TIME!
    total_capex = capex_electrolyzer_overnight+capex_compressor_installed+capex_storage_installed+capex_desal+capex_wind_installed+capex_solar_installed + capex_battery_installed #+h2_transport_CapEx#+ replacement_capex#capex_hybrid_installed
    #^modified
    
    #-------------------------------------- Add fixed costs--------------------------------
    pf.add_fixed_cost(name="Electrolyzer Fixed O&M Cost",usage=1.0,unit='$/year',cost=fixed_cost_electrolysis_total,escalation=gen_inflation)
    pf.add_fixed_cost(name="Desalination Fixed O&M Cost",usage=1.0,unit='$/year',cost=opex_desal,escalation=gen_inflation)
    #pf.add_fixed_cost(name="Hydrogen Transport Fixed O&M Cost",usage=1.0,unit='$/year',cost=h2_transport_OpEx,escalation=gen_inflation) #added
    
    pf.add_fixed_cost(name="Wind Plant Fixed O&M Cost",usage=1.0,unit='$/year',cost=fixed_cost_wind,escalation=gen_inflation)
    pf.add_fixed_cost(name="Solar Plant Fixed O&M Cost",usage=1.0,unit='$/year',cost=fixed_cost_solar,escalation=gen_inflation)
    pf.add_fixed_cost(name="Battery Storage Fixed O&M Cost",usage=1.0,unit='$/year',cost=fixed_cost_battery,escalation=gen_inflation)
        
    
    #---------------------- Add feedstocks, note the various cost options-------------------
    pf.add_feedstock(name='Water',usage=water_consumption_avg_galH2O_prkgH2,unit='gallon-water',cost=water_cost,escalation=gen_inflation)
    pf.add_feedstock(name='Var O&M',usage=1.0,unit='$/kg',cost=total_variable_OM_perkg,escalation=gen_inflation)
    
    pf.add_feedstock(name='Grid Electricity Cost',usage=grid_electricity_useage_kWhpkg,unit='$/kWh',cost=grid_price_perkWh,escalation=gen_inflation)
    #---------------------- Add various tax credit incentives -------------------
    pf.add_incentive(name ='Renewable PTC credit', value=Ren_PTC, decay = 0, sunset_years = Ren_PTC_duration, tax_credit = True)
    
    pf.add_incentive(name ='Hydrogen PTC credit', value=H2_PTC, decay = 0, sunset_years = H2_PTC_duration, tax_credit = True)
    
        
    sol = pf.solve_price()
    
    summary = pf.summary_vals
    
    price_breakdown = pf.get_cost_breakdown()

    # Calculate contribution of equipment to breakeven price
    total_price_capex = price_breakdown.loc[price_breakdown['Name']=='Electrolysis system','NPV'].tolist()[0]\
                      + price_breakdown.loc[price_breakdown['Name']=='Compression','NPV'].tolist()[0]\
                      + price_breakdown.loc[price_breakdown['Name']=='Hydrogen Storage','NPV'].tolist()[0]\
                      + price_breakdown.loc[price_breakdown['Name']=='Desalination','NPV'].tolist()[0]\
                      + price_breakdown.loc[price_breakdown['Name']=='Wind Plant','NPV'].tolist()[0]\
                      + price_breakdown.loc[price_breakdown['Name']=='Solar Plant','NPV'].tolist()[0]\
                      + price_breakdown.loc[price_breakdown['Name']=='Battery Storage','NPV'].tolist()[0]\
                      #+ price_breakdown.loc[price_breakdown['Name']=='Hydrogen Transport','NPV'].tolist()[0]\
                      

    capex_fraction = {'Electrolyzer':price_breakdown.loc[price_breakdown['Name']=='Electrolysis system','NPV'].tolist()[0]/total_price_capex,
                  'Compression':price_breakdown.loc[price_breakdown['Name']=='Compression','NPV'].tolist()[0]/total_price_capex,
                  'Hydrogen Storage':price_breakdown.loc[price_breakdown['Name']=='Hydrogen Storage','NPV'].tolist()[0]/total_price_capex,
                  'Desalination':price_breakdown.loc[price_breakdown['Name']=='Desalination','NPV'].tolist()[0]/total_price_capex,
                  'Wind Plant':price_breakdown.loc[price_breakdown['Name']=='Wind Plant','NPV'].tolist()[0]/total_price_capex,
                  'Solar Plant':price_breakdown.loc[price_breakdown['Name']=='Solar Plant','NPV'].tolist()[0]/total_price_capex,
                  'Battery Storage':price_breakdown.loc[price_breakdown['Name']=='Battery Storage','NPV'].tolist()[0]/total_price_capex,}
                  #'Hydrogen Transport':price_breakdown.loc[price_breakdown['Name']=='Hydrogen Transport','NPV'].tolist()[0]/total_price_capex} #<- added
    
    # Calculate financial expense associated with equipment
    cap_expense = price_breakdown.loc[price_breakdown['Name']=='Repayment of debt','NPV'].tolist()[0]\
        + price_breakdown.loc[price_breakdown['Name']=='Interest expense','NPV'].tolist()[0]\
        + price_breakdown.loc[price_breakdown['Name']=='Dividends paid','NPV'].tolist()[0]\
        - price_breakdown.loc[price_breakdown['Name']=='Inflow of debt','NPV'].tolist()[0]\
        - price_breakdown.loc[price_breakdown['Name']=='Inflow of equity','NPV'].tolist()[0]    
        
    # Calculate remaining financial expenses
    remaining_financial = price_breakdown.loc[price_breakdown['Name']=='Non-depreciable assets','NPV'].tolist()[0]\
        + price_breakdown.loc[price_breakdown['Name']=='Cash on hand reserve','NPV'].tolist()[0]\
        + price_breakdown.loc[price_breakdown['Name']=='Property insurance','NPV'].tolist()[0]\
        - price_breakdown.loc[price_breakdown['Name']=='Sale of non-depreciable assets','NPV'].tolist()[0]\
        - price_breakdown.loc[price_breakdown['Name']=='Cash on hand recovery','NPV'].tolist()[0]
    
    # Calculate LCOH breakdown and assign capital expense to equipment costs
    price_breakdown_electrolyzer = price_breakdown.loc[price_breakdown['Name']=='Electrolysis system','NPV'].tolist()[0] + cap_expense*capex_fraction['Electrolyzer']
    price_breakdown_compression = price_breakdown.loc[price_breakdown['Name']=='Compression','NPV'].tolist()[0] + cap_expense*capex_fraction['Compression']
    price_breakdown_storage = price_breakdown.loc[price_breakdown['Name']=='Hydrogen Storage','NPV'].tolist()[0]+cap_expense*capex_fraction['Hydrogen Storage']
    price_breakdown_desalination = price_breakdown.loc[price_breakdown['Name']=='Desalination','NPV'].tolist()[0] + cap_expense*capex_fraction['Desalination']
    price_breakdown_wind = price_breakdown.loc[price_breakdown['Name']=='Wind Plant','NPV'].tolist()[0] + cap_expense*capex_fraction['Wind Plant']
    price_breakdown_solar = price_breakdown.loc[price_breakdown['Name']=='Solar Plant','NPV'].tolist()[0] + cap_expense*capex_fraction['Solar Plant']
    price_breakdown_battery = price_breakdown.loc[price_breakdown['Name']=='Battery Storage','NPV'].tolist()[0] + cap_expense*capex_fraction['Battery Storage']
    #below was added
    #price_breakdown_h2_transport = price_breakdown.loc[price_breakdown['Name']=='Hydrogen Transport','NPV'].tolist()[0] + cap_expense*capex_fraction['Hydrogen Transport']

    
    price_breakdown_electrolysis_FOM = price_breakdown.loc[price_breakdown['Name']=='Electrolyzer Fixed O&M Cost','NPV'].tolist()[0]
    price_breakdown_electrolysis_VOM = price_breakdown.loc[price_breakdown['Name']=='Var O&M','NPV'].tolist()[0]
    price_breakdown_desalination_FOM = price_breakdown.loc[price_breakdown['Name']=='Desalination Fixed O&M Cost','NPV'].tolist()[0]
    price_breakdown_wind_FOM = price_breakdown.loc[price_breakdown['Name']=='Wind Plant Fixed O&M Cost','NPV'].tolist()[0]  
    price_breakdown_solar_FOM = price_breakdown.loc[price_breakdown['Name']=='Solar Plant Fixed O&M Cost','NPV'].tolist()[0]  
    price_breakdown_battery_FOM = price_breakdown.loc[price_breakdown['Name']=='Battery Storage Fixed O&M Cost','NPV'].tolist()[0]  
    #price_breakdown_h2_transport_FOM = price_breakdown.loc[price_breakdown['Name']=='Hydrogen Transport Fixed O&M Cost','NPV'].tolist()[0] #added
    price_breakdown_taxes = price_breakdown.loc[price_breakdown['Name']=='Income taxes payable','NPV'].tolist()[0]\
        - price_breakdown.loc[price_breakdown['Name'] == 'Monetized tax losses','NPV'].tolist()[0]\
            
    if gen_inflation > 0:
        price_breakdown_taxes = price_breakdown_taxes + price_breakdown.loc[price_breakdown['Name']=='Capital gains taxes payable','NPV'].tolist()[0]

    price_breakdown_water = price_breakdown.loc[price_breakdown['Name']=='Water','NPV'].tolist()[0]
    
    price_breakdown_grid_elec_price = price_breakdown.loc[price_breakdown['Name']=='Grid Electricity Cost','NPV'].tolist()[0]  
    
    price_breakdown_renewables=price_breakdown_wind + price_breakdown_solar +price_breakdown_battery
    price_breakdown_renewables_FOM = price_breakdown_wind_FOM + price_breakdown_solar_FOM + price_breakdown_battery_FOM
    lcoh_check = price_breakdown_electrolyzer+price_breakdown_compression+price_breakdown_storage+price_breakdown_electrolysis_FOM\
        + price_breakdown_desalination+price_breakdown_desalination_FOM+ price_breakdown_electrolysis_VOM\
            +price_breakdown_renewables+price_breakdown_renewables_FOM+price_breakdown_taxes+price_breakdown_water+price_breakdown_grid_elec_price+remaining_financial
               # + price_breakdown_h2_transport_FOM + price_breakdown_h2_transport
        
    lcoh_breakdown = {'LCOH: Compression & storage ($/kg)':price_breakdown_storage+price_breakdown_compression,\
                      'LCOH: Electrolyzer CAPEX ($/kg)':price_breakdown_electrolyzer,'LCOH: Desalination CAPEX ($/kg)':price_breakdown_desalination,\
                      'LCOH: Electrolyzer FOM ($/kg)':price_breakdown_electrolysis_FOM,'LCOH: Electrolyzer VOM ($/kg)':price_breakdown_electrolysis_VOM,\
                      'LCOH: Desalination FOM ($/kg)':price_breakdown_desalination_FOM,\
                      'LCOH: Wind Plant ($/kg)':price_breakdown_wind,'LCOH: Wind Plant FOM ($/kg)':price_breakdown_wind_FOM,\
                      'LCOH: Solar Plant ($/kg)':price_breakdown_solar,'LCOH: Solar Plant FOM ($/kg)':price_breakdown_solar_FOM,\
                      'LCOH: Battery Storage ($/kg)':price_breakdown_battery,'LCOH: Battery Storage FOM ($/kg)':price_breakdown_battery_FOM,\
                      'LCOH: Taxes ($/kg)':price_breakdown_taxes,\
                      'LCOH: Water consumption ($/kg)':price_breakdown_water,'LCOH: Grid electricity ($/kg)':price_breakdown_grid_elec_price,\
                      #'LCOH: Hydrogen Transport FOM ($/kg)':price_breakdown_h2_transport_FOM,'LCOH: Hydrogen Transport CAPEX ($/kg)':price_breakdown_h2_transport,\
                      'LCOH: Finances ($/kg)':remaining_financial,'LCOH: total ($/kg)':lcoh_check,'LCOH Profast:':sol['price']}
    
    price_breakdown = price_breakdown.drop(columns=['index','Amount'])
    # extra_outputs = {
    #     'electrolyzer_installed_cost_kw':capex_electrolyzer_overnight/electrolyzer_size_mw/1000,
    #     'elec_cf':elec_cf,
    #     # 'ren_frac':ren_frac,
    #     # 'electrolysis_total_EI_policy_grid':electrolysis_total_EI_policy_grid,
    #     # 'electrolysis_total_EI_policy_offgrid':electrolysis_total_EI_policy_offgrid,
    #     # 'H2_PTC':H2_PTC,
    #     'Ren_PTC':Ren_PTC,
    #     'h2_production_capex':total_capex
    # }
    # filepath = '/Users/egrant/Desktop/Electrolyzer-Repo/policy_invest/'
    # #filepath='/Users/egrant/Desktop/modular_hopp/'
    # pd.Series(dict(zip(summary['Name'],summary['Amount']))).to_pickle(filepath + 'summary_noITC_noH2PTC')
    # price_breakdown.to_pickle(filepath + 'pricebreakdown_noITC_noH2PTC')
    # return(sol,summary,price_breakdown,lcoh_breakdown)#,extra_outputs)
    # return(sol,summary,price_breakdown,lcoh_breakdown)
    capex_df_keys = ['Electrolysis system [$]','Compression [$]','Hydrogen Storage [$]','Desalination [$]','Wind Plant [$]','Solar Plant [$]','Battery [$]']
    capex_df_vals=[price_breakdown.loc[price_breakdown['Name']=='Electrolysis system','NPV'].tolist()[0],\
                    price_breakdown.loc[price_breakdown['Name']=='Compression','NPV'].tolist()[0],\
                    price_breakdown.loc[price_breakdown['Name']=='Hydrogen Storage','NPV'].tolist()[0],\
                    price_breakdown.loc[price_breakdown['Name']=='Desalination','NPV'].tolist()[0],\
                    price_breakdown.loc[price_breakdown['Name']=='Wind Plant','NPV'].tolist()[0],\
                    price_breakdown.loc[price_breakdown['Name']=='Solar Plant','NPV'].tolist()[0],\
                    price_breakdown.loc[price_breakdown['Name']=='Battery Storage','NPV'].tolist()[0]]
    capex_df = pd.Series(dict(zip(capex_df_keys,capex_df_vals)))
    return(sol,lcoh_breakdown,capex_df)