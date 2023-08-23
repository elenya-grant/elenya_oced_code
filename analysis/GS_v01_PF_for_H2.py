
import sys
#sys.path.insert(1,'../PyFAST/')
import numpy as np
import pandas as pd
profast_dir = '/Users/egrant/Desktop/HOPP-GIT/HOPP/ProFAST-main/' #user must specify!
sys.path.insert(1,profast_dir)
#sys.path.insert(1,'/Users/egrant/Desktop/HOPP-GIT/HOPP/ProFAST-main/')
import ProFAST

# from optimization.gradient_opt_esg import simple_opt
#from analysis.LCA_single_scenario_ProFAST import hydrogen_LCA_singlescenario_ProFAST

#sys.path.append('../ProFAST/')

pf = ProFAST.ProFAST()
class LCOH_Calc:
    def __init__(self,config,cost_year,policy_desc,re_cost_desc):
        self.H2_PTC_duration = config["policy_cases"]["H2_PTC_duration"]
        self.Ren_PTC_duration = config["policy_cases"]["Ren_PTC_duration"]
        
        self.use_solar_ITC=config["policy_cases"][policy_desc]["use_solar_ITC"]
        self.use_solar_PTC=config["policy_cases"][policy_desc]["use_solar_PTC"]
        

        self.gen_inflation=config["finance_parameters"]["general_inflation"]
        self.debt_type = config["finance_parameters"]["debt_type"]
        self.depreciation_method = config["finance_parameters"]["depreciation_method"]
        self.depreciation_period = config["finance_parameters"]["depreciation_period"]
        #TODO: replace hard-coded depreciation period with ^
        self.depreciation_period_electrolyzer = config["finance_parameters"]["depreciation_period_electrolyzer"]
        
        self.kg_water_pr_kg_H2 = config["electrolyzer"]["config"]["kg_H20_pr_kg_H2"]
        self.policy_info = config["policy_cases"][policy_desc]
        self.re_plant_cost_info = config["renewable_cost_cases"][cost_year][re_cost_desc]

        #below is okay to be hard-coded for now... probably
        self.plant_life = config["plant_general"]["plant_life"]
        self.water_cost_USD_pr_gal = config["cost_info"]["water_cost_USD_pr_gal"]
        pem_cost_params = config["cost_info"]["electrolyzer"]

        self.stack_replacement_perc = pem_cost_params["electrolysis_cost_params"]["stack_replacement_cost"]
        self.pem_VOM_pr_MWh = pem_cost_params["electrolysis_cost_params"]["VOM_USD_pr_MWh"]
        self.pem_install_fac = pem_cost_params["electrolysis_cost_params"]["electrolyzer_installation_factor"]
        self.pem_FOM_pr_kW = pem_cost_params["electrolysis_cost_params"]["FOM_USD_pr_kW"]

        self.pem_indirect_cost_perc = (pem_cost_params["indirect_CapEx_Costs"]["site_prep"] + 
        pem_cost_params["indirect_CapEx_Costs"]["project_contingency"] +
        pem_cost_params["indirect_CapEx_Costs"]["engineering_design"] +
        pem_cost_params["indirect_CapEx_Costs"]["permitting"])

        self.desal_OpEx_pr_kgprsec = pem_cost_params["desal_cost_params"]["desal_OpEx_USD_pr_kg_pr_sec"]
        self.desal_CapEx_pr_kgprsec =pem_cost_params["desal_cost_params"]["desal_CapEx_USD_pr_kg_pr_sec"]

        self.compressor_capex_pr_kWelec = config["hydrogen_storage_cases"]["compressor_CapEx_USD_pr_kW"]
        self.compressor_opex_pr_kWelec =config["hydrogen_storage_cases"]["compressor_OpEx_USD_pr_kW"]
        # pem_cost_params[""]
        # pem_cost_params[""]

        self.elec_design_eff_kWh_pr_kg = config["electrolyzer"]["config"]['kWh_pr_kg_design']
        
        pass
    def run_lcoh2_storage(self,pf,elec_cf,electrolyzer_size_MW,hydrogen_storage_size_kg,hydrogen_storage_capex_pr_kg,hydrogen_storage_opex_pr_kg):
        
        # pf.set_params('non depr assets',0)
        # pf.set_params('end of proj sale non depr assets',0)
        # pf.set_params('sales tax',0) 
        # pf.set_params('property tax and insurance percent',0)
        #add usage params
        #print('LCOH H2 Storage:')
        pf = self.add_electrolyzer_usage_params(elec_cf,electrolyzer_size_MW,pf)
        #add hydrogen storage ITC ... hmm...
        pf,_ = self.add_ITC_for_hydrogen_storage(hydrogen_storage_size_kg,hydrogen_storage_capex_pr_kg,pf)
        #set capital costs
        pf = self.add_hydrogen_storage_capital_costs(electrolyzer_size_MW,hydrogen_storage_size_kg,hydrogen_storage_capex_pr_kg,pf)
        #add O&M costs
        pf = self.add_hydrogen_storage_FOM_costs(electrolyzer_size_MW,hydrogen_storage_size_kg,hydrogen_storage_opex_pr_kg,pf)
        #add
        sol,summary,price_breakdown,lcoh_breakdown =self.get_lcoh2_storage_outputs(pf)
        return sol,summary,price_breakdown,lcoh_breakdown
    
    def run_lcoh_nostorage(self,pf,component_sizes,elec_avg_consumption_kWhprkg,elec_cf,annual_hydrogen_kg,avg_stack_life_hrs,wind_frac):
        # self.policy_info['itc']
        # self.policy_info['h2_ptc']
        # self.policy_info['Ren_PTC_mult']
        #TODO: make renewable_plant_cost_info_kWh_pr_kg based only on wind!
        wind_size_mw = component_sizes["wind_size_mw"]
        solar_size_mw = component_sizes["solar_size_mw"]
        battery_size_mw = component_sizes["battery_size_mw"]
        battery_hrs = component_sizes["battery_hrs"]
        electrolyzer_size_MW = component_sizes["electrolyzer_size_mw"]
        desal_size_kg_pr_sec = component_sizes["desal_size_kg_pr_sec"]
        renewable_plant_cost_info = self.re_plant_cost_info
        electrolyzer_capex_pr_kW = renewable_plant_cost_info["electrolyzer"]["uninstalled_CapEx_pr_kW"]
        #finishing setting params
        pf = self.add_electrolyzer_usage_params(elec_cf,electrolyzer_size_MW,pf)
        pf,_ = self.add_RE_ITC_policy(renewable_plant_cost_info,solar_size_mw,battery_size_mw,battery_hrs,pf)
        #add capital costs

        electrolyzer_refurbishment_schedule = self.make_refurbishment_scehdule(avg_stack_life_hrs)

        pf = self.add_re_plant_capital_costs(renewable_plant_cost_info,wind_size_mw,solar_size_mw,battery_size_mw,battery_hrs,pf)
        pf = self.add_electrolyzer_capital_costs(electrolyzer_capex_pr_kW,electrolyzer_size_MW,desal_size_kg_pr_sec,electrolyzer_refurbishment_schedule,pf) #TODO: finish

        #add fixed costs
        pf = self.add_re_plant_FOM(renewable_plant_cost_info,wind_size_mw,solar_size_mw,battery_size_mw,battery_hrs,pf)
        pf = self.add_electrolyzer_FOM(electrolyzer_size_MW,desal_size_kg_pr_sec,pf)
        #add feedstock costs
        pf = self.add_feedstock_costs(annual_hydrogen_kg,elec_avg_consumption_kWhprkg,pf)

        #add policy
        # pf = self.add_Ren_PTC(ren_energy_usage_kWh_pr_kg,pf)
        if self.use_solar_PTC:
            pf = self.add_Ren_PTC(elec_avg_consumption_kWhprkg,pf)
        else:
            ren_energy_usage_kWh_pr_kg = wind_frac*elec_avg_consumption_kWhprkg
            pf = self.add_Ren_PTC(ren_energy_usage_kWh_pr_kg,pf)
        h2_ptc = self.policy_info['h2_ptc']
        pf = self.add_H2_PTC(h2_ptc,pf)

        sol,summary,price_breakdown,lcoh_breakdown = self.get_outputs_lcoh_nostorage(pf)

        return sol,summary,price_breakdown,lcoh_breakdown
    def run_lcoh_full(self,pf,component_sizes,elec_avg_consumption_kWhprkg,elec_cf,annual_hydrogen_kg,avg_stack_life_hrs,wind_frac,hydrogen_storage_capex_pr_kg,hydrogen_storage_opex_pr_kg):
        wind_size_mw = component_sizes["wind_size_mw"]
        solar_size_mw = component_sizes["solar_size_mw"]
        battery_size_mw = component_sizes["battery_size_mw"]
        battery_hrs = component_sizes["battery_hrs"]
        electrolyzer_size_MW = component_sizes["electrolyzer_size_mw"]
        desal_size_kg_pr_sec = component_sizes["desal_size_kg_pr_sec"]
        hydrogen_storage_size_kg = component_sizes['hydrogen_storage_size_kg']
        renewable_plant_cost_info = self.re_plant_cost_info
        electrolyzer_capex_pr_kW = renewable_plant_cost_info["electrolyzer"]["uninstalled_CapEx_pr_kW"]
        #finishing setting params
        pf = self.add_electrolyzer_usage_params(elec_cf,electrolyzer_size_MW,pf)
        pf,ren_itc = self.add_RE_ITC_policy(renewable_plant_cost_info,solar_size_mw,battery_size_mw,battery_hrs,pf)
        pf,_ = self.add_ITC_for_hydrogen_storage(hydrogen_storage_size_kg,hydrogen_storage_capex_pr_kg,pf,re_itc = ren_itc)
        
        #add capital costs

        
        electrolyzer_refurbishment_schedule = self.make_refurbishment_scehdule(avg_stack_life_hrs)
        
        pf = self.add_re_plant_capital_costs(renewable_plant_cost_info,wind_size_mw,solar_size_mw,battery_size_mw,battery_hrs,pf)
        pf = self.add_electrolyzer_capital_costs(electrolyzer_capex_pr_kW,electrolyzer_size_MW,desal_size_kg_pr_sec,electrolyzer_refurbishment_schedule,pf) #TODO: finish
        pf = self.add_hydrogen_storage_capital_costs(electrolyzer_size_MW,hydrogen_storage_size_kg,hydrogen_storage_capex_pr_kg,pf)
        #add fixed costs
        pf = self.add_re_plant_FOM(renewable_plant_cost_info,wind_size_mw,solar_size_mw,battery_size_mw,battery_hrs,pf)
        pf = self.add_electrolyzer_FOM(electrolyzer_size_MW,desal_size_kg_pr_sec,pf)
        pf = self.add_hydrogen_storage_FOM_costs(electrolyzer_size_MW,hydrogen_storage_size_kg,hydrogen_storage_opex_pr_kg,pf)
        #add feedstock costs
        pf = self.add_feedstock_costs(annual_hydrogen_kg,elec_avg_consumption_kWhprkg,pf)

        #add policy
        #TODO: fix this!
        if self.use_solar_PTC:
            #double-dipping, all power to electrolyzer is from wind and solar
            pf = self.add_Ren_PTC(elec_avg_consumption_kWhprkg,pf)
        else:
            #
            ren_energy_usage_kWh_pr_kg = wind_frac*elec_avg_consumption_kWhprkg
            pf = self.add_Ren_PTC(ren_energy_usage_kWh_pr_kg,pf)
            # pf = self.add_Ren_PTC(elec_avg_consumption_kWhprkg,pf)    
        # pf = self.add_Ren_PTC(ren_energy_usage_kWh_pr_kg,pf)
        h2_ptc = self.policy_info['h2_ptc']
        pf = self.add_H2_PTC(h2_ptc,pf)

        sol,summary,price_breakdown,lcoh_breakdown = self.get_outputs_full_lcoh(pf)

        return sol,summary,price_breakdown,lcoh_breakdown
    def make_refurbishment_scehdule(self,avg_stack_life_hrs):
        if not np.isnan(avg_stack_life_hrs):
            refturb_period = np.max([1,int(np.floor(avg_stack_life_hrs/(24*365)))])

            electrolyzer_refurbishment_schedule = np.zeros(self.plant_life)
            electrolyzer_refurbishment_schedule[refturb_period:self.plant_life:refturb_period]=self.stack_replacement_perc
        else:
            electrolyzer_refurbishment_schedule = []
        return electrolyzer_refurbishment_schedule
    def add_electrolyzer_usage_params(self,elec_cf,electrolyzer_size_MW,pf):
        electrolysis_plant_capacity_kgperday=   electrolyzer_size_MW/self.elec_design_eff_kWh_pr_kg*1000*24
        pf.set_params('capacity',electrolysis_plant_capacity_kgperday) 
        pf.set_params('long term utilization',elec_cf)
        return pf
    # def add_ITC_policy(self,ITC,pf):
    #     capex_for_itc = capex_solar_installed + capex_battery_installed + 
    #     pf.set_params('one time cap inct',{'value':np.nan_to_num(ITC*capex_solar_installed),'depr type':'MACRS','depr period':7,'depreciable':True})
    def add_RE_ITC_policy(self,renewable_plant_cost_info,solar_size_mw,battery_size_mw,battery_hrs,pf):
        if self.use_solar_ITC:
            # capex_solar_installed = solar_size_mw*1000*renewable_plant_cost_info['solar']['capex_per_kw']
            capex_solar_installed = solar_size_mw*1000*renewable_plant_cost_info['pv']['capex_per_kw']
        else: 
            capex_solar_installed = 0 
        battery_cost_pr_kW = renewable_plant_cost_info['battery']['capex_per_kw'] + (renewable_plant_cost_info['battery']['capex_per_kwh']*battery_hrs)
        capex_battery_installed = battery_size_mw*1000*battery_cost_pr_kW
        itc_amt = self.policy_info['itc']*(capex_solar_installed + capex_battery_installed)
        pf.set_params('one time cap inct',{'value':itc_amt,'depr type':self.depreciation_method,'depr period':7,'depreciable':True})
        # pf.set_params('one time cap inct',{'value':np.nan_to_num(ITC*capex_solar_installed),'depr type':'MACRS','depr period':7,'depreciable':True})
        # pf.set_params('one time cap inct',{'value':np.nan_to_num(ITC*capex_battery_installed),'depr type':'MACRS','depr period':7,'depreciable':True})
        return pf,itc_amt
    # def add_hydrogen_storage_ITC(self,itc,pf):
    #     pf.set_params('one time cap inct',{'value':np.nan_to_num(itc*capex_storage_installed),'depr type':'MACRS','depr period':7,'depreciable':True})
    #     return pf
    def add_Ren_PTC(self,ren_energy_usage_kWh_pr_kg,pf):
        # self.policy_info['Ren_PTC_mult']
        Ren_PTC = self.policy_info['Ren_PTC_mult']*ren_energy_usage_kWh_pr_kg
        pf.add_incentive(name ='Renewable PTC credit', value=Ren_PTC, decay = 0, sunset_years = self.Ren_PTC_duration, tax_credit = True)


        return pf
    def add_H2_PTC(self,h2_ptc,pf):
        
        pf.add_incentive(name ='Hydrogen PTC credit', value=h2_ptc, decay = 0, sunset_years = self.H2_PTC_duration, tax_credit = True)

        return pf
    
    def add_re_plant_capital_costs(self,renewable_plant_cost_info,wind_size_mw,solar_size_mw,battery_size_mw,battery_hrs,pf):
        capex_wind_installed = wind_size_mw*1000*renewable_plant_cost_info['wind']['capex_per_kw']
        # capex_solar_installed = solar_size_mw*1000*renewable_plant_cost_info['solar']['capex_per_kw']
        capex_solar_installed = solar_size_mw*1000*renewable_plant_cost_info['pv']['capex_per_kw']
        battery_cost_pr_kW = renewable_plant_cost_info['battery']['capex_per_kw'] + (renewable_plant_cost_info['battery']['capex_per_kwh']*battery_hrs)
        capex_battery_installed = battery_size_mw*1000*battery_cost_pr_kW
        pf.add_capital_item(name = "Wind Plant",cost = capex_wind_installed,depr_type = self.depreciation_method,depr_period = 7,refurb = [0])
        pf.add_capital_item(name = "Solar Plant",cost = capex_solar_installed,depr_type = self.depreciation_method,depr_period = 7,refurb = [0])
        pf.add_capital_item(name = "Battery Storage",cost = capex_battery_installed,depr_type = self.depreciation_method,depr_period = 7,refurb = [0])
        return pf
    def add_electrolyzer_capital_costs(self,electrolyzer_capex_pr_kW,electrolyzer_size_MW,desal_size_kg_pr_sec,electrolyzer_refurbishment_schedule,pf):
        direct_capex_pr_kW = electrolyzer_capex_pr_kW*(1+self.pem_install_fac)
        indirect_capex_pr_kW = direct_capex_pr_kW*self.pem_indirect_cost_perc
        capex_electrolyzer_overnight_pr_kW = direct_capex_pr_kW + indirect_capex_pr_kW
        capex_electrolyzer_overnight = electrolyzer_size_MW * 1000 * capex_electrolyzer_overnight_pr_kW
        
        capex_compressor_installed = electrolyzer_size_MW * 1000 * self.compressor_capex_pr_kWelec

        capex_desal = desal_size_kg_pr_sec*self.desal_CapEx_pr_kgprsec
        pf.add_capital_item(name="Electrolysis system",cost=capex_electrolyzer_overnight,depr_type=self.depreciation_method,depr_period=7,refurb=list(electrolyzer_refurbishment_schedule))
        pf.add_capital_item(name ="Desalination",cost = capex_desal,depr_type=self.depreciation_method,depr_period=7,refurb=[0])
        pf.add_capital_item(name="Compression",cost=capex_compressor_installed,depr_type=self.depreciation_method,depr_period=7,refurb=[0])

        return pf
    def add_ITC_for_hydrogen_storage(self,hydrogen_storage_size_kg,hydrogen_storage_capex_pr_kg,pf,re_itc=0):
        itc_amt = re_itc + (self.policy_info['itc']*(hydrogen_storage_size_kg*hydrogen_storage_capex_pr_kg))
        pf.set_params('one time cap inct',{'value':itc_amt,'depr type':self.depreciation_method,'depr period':7,'depreciable':True})
        return pf,itc_amt
    def add_hydrogen_storage_capital_costs(self,electrolyzer_size_MW,hydrogen_storage_size_kg,hydrogen_storage_capex_pr_kg,pf):
        #compression & storage
        # capex_compressor_installed = self.compressor_capex_pr_kWelec*electrolyzer_size_MW*1000
        capex_storage_installed = hydrogen_storage_size_kg*hydrogen_storage_capex_pr_kg
        # pf.add_capital_item(name="Compression",cost=capex_compressor_installed,depr_type=self.depreciation_method,depr_period=7,refurb=[0])
        pf.add_capital_item(name="Hydrogen Storage",cost=capex_storage_installed,depr_type=self.depreciation_method,depr_period=7,refurb=[0])
        return pf

    def add_hydrogen_storage_FOM_costs(self,electrolyzer_size_MW,hydrogen_storage_size_kg,hydrogen_storage_opex_pr_kg,pf):
        #TODO: double check these!
        #compression & storage
        # fixed_cost_compressor = self.compressor_opex_pr_kWelec*electrolyzer_size_MW*1000
        # fixed_cost_compressor = compressor_opex_pr_kW*electrolyzer_size_MW*1000
        fixed_cost_hydrogen_storage = hydrogen_storage_opex_pr_kg*hydrogen_storage_size_kg
        pf.add_fixed_cost(name="Hydrogen Storage Fixed O&M Cost",usage=1.0,unit='$/year',cost=fixed_cost_hydrogen_storage,escalation=self.gen_inflation)
        # pf.add_fixed_cost(name="Compression Fixed O&M Cost",usage=1.0,unit='$/year',cost=fixed_cost_compressor,escalation=self.gen_inflation)
        
        return pf
    def add_re_plant_FOM(self,renewable_plant_cost_info,wind_size_mw,solar_size_mw,battery_size_mw,battery_hrs,pf):
        #wind, solar, battery
        fixed_cost_wind = wind_size_mw*1000*renewable_plant_cost_info['wind']['o&m_per_kw']
        # fixed_cost_solar = solar_size_mw*1000*renewable_plant_cost_info['solar']['o&m_per_kw']
        fixed_cost_solar = solar_size_mw*1000*renewable_plant_cost_info['pv']['o&m_per_kw']
        
        # renewable_plant_cost_info['battery']['capex_per_kw']
        # renewable_plant_cost_info['battery']['capex_per_kwh']
        battery_cost_pr_kW = renewable_plant_cost_info['battery']['capex_per_kw'] + (renewable_plant_cost_info['battery']['capex_per_kwh']*battery_hrs)
        capex_battery_installed = battery_size_mw*1000*battery_cost_pr_kW
        fixed_cost_battery=capex_battery_installed*renewable_plant_cost_info['battery']['o&m_percent']
        pf.add_fixed_cost(name="Wind Plant Fixed O&M Cost",usage=1.0,unit='$/year',cost=fixed_cost_wind,escalation=self.gen_inflation)
        pf.add_fixed_cost(name="Solar Plant Fixed O&M Cost",usage=1.0,unit='$/year',cost=fixed_cost_solar,escalation=self.gen_inflation)
        pf.add_fixed_cost(name="Battery Storage Fixed O&M Cost",usage=1.0,unit='$/year',cost=fixed_cost_battery,escalation=self.gen_inflation)
            
        return pf
    def add_electrolyzer_FOM(self,electrolyzer_size_MW,desal_size_kg_pr_sec,pf):
        fixed_cost_electrolysis_total = self.pem_FOM_pr_kW*electrolyzer_size_MW * 1000
        opex_desal = self.desal_OpEx_pr_kgprsec * desal_size_kg_pr_sec
        fixed_cost_compressor = self.compressor_opex_pr_kWelec*electrolyzer_size_MW*1000
        pf.add_fixed_cost(name="Electrolyzer Fixed O&M Cost",usage=1.0,unit='$/year',cost=fixed_cost_electrolysis_total,escalation=self.gen_inflation)
        pf.add_fixed_cost(name="Desalination Fixed O&M Cost",usage=1.0,unit='$/year',cost=opex_desal,escalation=self.gen_inflation)
        pf.add_fixed_cost(name="Compression Fixed O&M Cost",usage=1.0,unit='$/year',cost=fixed_cost_compressor,escalation=self.gen_inflation)

        #desal + electrolyzer
        return pf
    def add_feedstock_costs(self,annual_hydrogen_kg,elec_avg_consumption_kWhprkg,pf):
        # water_consumption_avg_galH2O_prkgH2 = self.kg_water_pr_kg_H2*annual_hydrogen_kg/3.79 #kg->gal
        water_consumption_avg_galH2O_prkgH2 = self.kg_water_pr_kg_H2/3.79 #kg->gal

        total_variable_OM_perkg = self.pem_VOM_pr_MWh*elec_avg_consumption_kWhprkg/1000
        pf.add_feedstock(name='Water',usage=water_consumption_avg_galH2O_prkgH2,unit='gallon-water',cost=self.water_cost_USD_pr_gal,escalation=self.gen_inflation)
        pf.add_feedstock(name='Var O&M',usage=1.0,unit='$/kg',cost=total_variable_OM_perkg,escalation=self.gen_inflation)
    
        return pf
    def get_outputs_lcoh_nostorage(self,pf):
        sol = pf.solve_price()
    
        summary = pf.summary_vals
        
        price_breakdown = pf.get_cost_breakdown()

        total_price_capex = price_breakdown.loc[price_breakdown['Name']=='Electrolysis system','NPV'].tolist()[0]\
                      + price_breakdown.loc[price_breakdown['Name']=='Desalination','NPV'].tolist()[0]\
                      + price_breakdown.loc[price_breakdown['Name']=='Wind Plant','NPV'].tolist()[0]\
                      + price_breakdown.loc[price_breakdown['Name']=='Solar Plant','NPV'].tolist()[0]\
                      + price_breakdown.loc[price_breakdown['Name']=='Battery Storage','NPV'].tolist()[0]\
                    #     + price_breakdown.loc[price_breakdown['Name']=='Compression','NPV'].tolist()[0]\
                    #   + price_breakdown.loc[price_breakdown['Name']=='Hydrogen Storage','NPV'].tolist()[0]\
        
        capex_fraction = {'Electrolyzer':price_breakdown.loc[price_breakdown['Name']=='Electrolysis system','NPV'].tolist()[0]/total_price_capex,
                    'Compression':price_breakdown.loc[price_breakdown['Name']=='Compression','NPV'].tolist()[0]/total_price_capex,
                    # 'Hydrogen Storage':price_breakdown.loc[price_breakdown['Name']=='Hydrogen Storage','NPV'].tolist()[0]/total_price_capex,
                    'Desalination':price_breakdown.loc[price_breakdown['Name']=='Desalination','NPV'].tolist()[0]/total_price_capex,
                    'Wind Plant':price_breakdown.loc[price_breakdown['Name']=='Wind Plant','NPV'].tolist()[0]/total_price_capex,
                    'Solar Plant':price_breakdown.loc[price_breakdown['Name']=='Solar Plant','NPV'].tolist()[0]/total_price_capex,
                    'Battery Storage':price_breakdown.loc[price_breakdown['Name']=='Battery Storage','NPV'].tolist()[0]/total_price_capex,}

        
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
        # price_breakdown_storage = price_breakdown.loc[price_breakdown['Name']=='Hydrogen Storage','NPV'].tolist()[0]+cap_expense*capex_fraction['Hydrogen Storage']
        price_breakdown_desalination = price_breakdown.loc[price_breakdown['Name']=='Desalination','NPV'].tolist()[0] + cap_expense*capex_fraction['Desalination']
        price_breakdown_wind = price_breakdown.loc[price_breakdown['Name']=='Wind Plant','NPV'].tolist()[0] + cap_expense*capex_fraction['Wind Plant']
        price_breakdown_solar = price_breakdown.loc[price_breakdown['Name']=='Solar Plant','NPV'].tolist()[0] + cap_expense*capex_fraction['Solar Plant']
        price_breakdown_battery = price_breakdown.loc[price_breakdown['Name']=='Battery Storage','NPV'].tolist()[0] + cap_expense*capex_fraction['Battery Storage']
        
        price_breakdown_electrolysis_FOM = price_breakdown.loc[price_breakdown['Name']=='Electrolyzer Fixed O&M Cost','NPV'].tolist()[0]
        price_breakdown_electrolysis_VOM = price_breakdown.loc[price_breakdown['Name']=='Var O&M','NPV'].tolist()[0]
        price_breakdown_desalination_FOM = price_breakdown.loc[price_breakdown['Name']=='Desalination Fixed O&M Cost','NPV'].tolist()[0]
        price_breakdown_wind_FOM = price_breakdown.loc[price_breakdown['Name']=='Wind Plant Fixed O&M Cost','NPV'].tolist()[0]  
        price_breakdown_solar_FOM = price_breakdown.loc[price_breakdown['Name']=='Solar Plant Fixed O&M Cost','NPV'].tolist()[0]  
        price_breakdown_battery_FOM = price_breakdown.loc[price_breakdown['Name']=='Battery Storage Fixed O&M Cost','NPV'].tolist()[0]  
        
        price_breakdown_compressor_FOM = price_breakdown.loc[price_breakdown['Name']=='Compression Fixed O&M Cost','NPV'].tolist()[0]  
        # price_breakdown_h2_transport_FOM = price_breakdown.loc[price_breakdown['Name']=='Hydrogen Transport Fixed O&M Cost','NPV'].tolist()[0] #added
        price_breakdown_taxes = price_breakdown.loc[price_breakdown['Name']=='Income taxes payable','NPV'].tolist()[0]\
            - price_breakdown.loc[price_breakdown['Name'] == 'Monetized tax losses','NPV'].tolist()[0]\
                
        if self.gen_inflation > 0:
            price_breakdown_taxes = price_breakdown_taxes + price_breakdown.loc[price_breakdown['Name']=='Capital gains taxes payable','NPV'].tolist()[0]

        price_breakdown_water = price_breakdown.loc[price_breakdown['Name']=='Water','NPV'].tolist()[0]
        
        price_breakdown_grid_elec_price = 0#price_breakdown.loc[price_breakdown['Name']=='Grid Electricity Cost','NPV'].tolist()[0]  
        if self.policy_info['itc']>0:
            price_breakdown_ITC = price_breakdown.loc[price_breakdown['Name']=='One time capital incentive','NPV'].tolist()[0]
        else:
            price_breakdown_ITC = 0

        price_breakdown_renewables=price_breakdown_wind + price_breakdown_solar +price_breakdown_battery

        price_breakdown_renewables_FOM = price_breakdown_wind_FOM + price_breakdown_solar_FOM + price_breakdown_battery_FOM
        lcoh_check = price_breakdown_electrolyzer+price_breakdown_electrolysis_FOM + price_breakdown_compression + price_breakdown_compressor_FOM\
            + price_breakdown_desalination+price_breakdown_desalination_FOM+ price_breakdown_electrolysis_VOM\
                +price_breakdown_renewables+price_breakdown_renewables_FOM+price_breakdown_taxes+price_breakdown_water+price_breakdown_grid_elec_price+remaining_financial\
                    -price_breakdown_ITC
        lcoh_breakdown = {'LCOH: Electrolyzer CAPEX ($/kg)':price_breakdown_electrolyzer,'LCOH: Desalination CAPEX ($/kg)':price_breakdown_desalination,\
                        'LCOH: Compressor ($/kg)':price_breakdown_compression,\
                        'LCOH: Electrolyzer FOM ($/kg)':price_breakdown_electrolysis_FOM,'LCOH: Electrolyzer VOM ($/kg)':price_breakdown_electrolysis_VOM,\
                        'LCOH: Desalination FOM ($/kg)':price_breakdown_desalination_FOM,\
                        'LCOH: Compressor FOM ($/kg)':price_breakdown_compressor_FOM,\
                        'LCOH: Wind Plant ($/kg)':price_breakdown_wind,'LCOH: Wind Plant FOM ($/kg)':price_breakdown_wind_FOM,\
                        'LCOH: Solar Plant ($/kg)':price_breakdown_solar,'LCOH: Solar Plant FOM ($/kg)':price_breakdown_solar_FOM,\
                        'LCOH: Battery Storage ($/kg)':price_breakdown_battery,'LCOH: Battery Storage FOM ($/kg)':price_breakdown_battery_FOM,\
                        'LCOH: Taxes ($/kg)':price_breakdown_taxes,\
                        'LCOH: Water consumption ($/kg)':price_breakdown_water,'LCOH: Grid electricity ($/kg)':price_breakdown_grid_elec_price,\
                        'LCOH: ITC ($/kg)':-1*price_breakdown_ITC,\
                        'LCOH: Finances ($/kg)':remaining_financial,'LCOH: total ($/kg)':lcoh_check,'LCOH Profast:':sol['price']}
        
        return(sol,summary,price_breakdown,lcoh_breakdown)
    def get_outputs_full_lcoh(self,pf):
        sol = pf.solve_price()
    
        summary = pf.summary_vals
        
        price_breakdown = pf.get_cost_breakdown()

        total_price_capex = price_breakdown.loc[price_breakdown['Name']=='Electrolysis system','NPV'].tolist()[0]\
                      + price_breakdown.loc[price_breakdown['Name']=='Compression','NPV'].tolist()[0]\
                      + price_breakdown.loc[price_breakdown['Name']=='Hydrogen Storage','NPV'].tolist()[0]\
                      + price_breakdown.loc[price_breakdown['Name']=='Desalination','NPV'].tolist()[0]\
                      + price_breakdown.loc[price_breakdown['Name']=='Wind Plant','NPV'].tolist()[0]\
                      + price_breakdown.loc[price_breakdown['Name']=='Solar Plant','NPV'].tolist()[0]\
                      + price_breakdown.loc[price_breakdown['Name']=='Battery Storage','NPV'].tolist()[0]\
        
        capex_fraction = {'Electrolyzer':price_breakdown.loc[price_breakdown['Name']=='Electrolysis system','NPV'].tolist()[0]/total_price_capex,
                    'Compression':price_breakdown.loc[price_breakdown['Name']=='Compression','NPV'].tolist()[0]/total_price_capex,
                    'Hydrogen Storage':price_breakdown.loc[price_breakdown['Name']=='Hydrogen Storage','NPV'].tolist()[0]/total_price_capex,
                    'Desalination':price_breakdown.loc[price_breakdown['Name']=='Desalination','NPV'].tolist()[0]/total_price_capex,
                    'Wind Plant':price_breakdown.loc[price_breakdown['Name']=='Wind Plant','NPV'].tolist()[0]/total_price_capex,
                    'Solar Plant':price_breakdown.loc[price_breakdown['Name']=='Solar Plant','NPV'].tolist()[0]/total_price_capex,
                    'Battery Storage':price_breakdown.loc[price_breakdown['Name']=='Battery Storage','NPV'].tolist()[0]/total_price_capex,}

        
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
        
        price_breakdown_electrolysis_FOM = price_breakdown.loc[price_breakdown['Name']=='Electrolyzer Fixed O&M Cost','NPV'].tolist()[0]
        price_breakdown_electrolysis_VOM = price_breakdown.loc[price_breakdown['Name']=='Var O&M','NPV'].tolist()[0]
        price_breakdown_desalination_FOM = price_breakdown.loc[price_breakdown['Name']=='Desalination Fixed O&M Cost','NPV'].tolist()[0]
        price_breakdown_wind_FOM = price_breakdown.loc[price_breakdown['Name']=='Wind Plant Fixed O&M Cost','NPV'].tolist()[0]  
        price_breakdown_solar_FOM = price_breakdown.loc[price_breakdown['Name']=='Solar Plant Fixed O&M Cost','NPV'].tolist()[0]  
        price_breakdown_battery_FOM = price_breakdown.loc[price_breakdown['Name']=='Battery Storage Fixed O&M Cost','NPV'].tolist()[0]  

        price_breakdown_storage_FOM = price_breakdown.loc[price_breakdown['Name']=='Hydrogen Storage Fixed O&M Cost','NPV'].tolist()[0]  
        price_breakdown_compressor_FOM = price_breakdown.loc[price_breakdown['Name']=='Compression Fixed O&M Cost','NPV'].tolist()[0]  
        # price_breakdown_h2_transport_FOM = price_breakdown.loc[price_breakdown['Name']=='Hydrogen Transport Fixed O&M Cost','NPV'].tolist()[0] #added
        price_breakdown_taxes = price_breakdown.loc[price_breakdown['Name']=='Income taxes payable','NPV'].tolist()[0]\
            - price_breakdown.loc[price_breakdown['Name'] == 'Monetized tax losses','NPV'].tolist()[0]\
                
        if self.gen_inflation > 0:
            price_breakdown_taxes = price_breakdown_taxes + price_breakdown.loc[price_breakdown['Name']=='Capital gains taxes payable','NPV'].tolist()[0]

        price_breakdown_water = price_breakdown.loc[price_breakdown['Name']=='Water','NPV'].tolist()[0]
        
        price_breakdown_grid_elec_price = 0#price_breakdown.loc[price_breakdown['Name']=='Grid Electricity Cost','NPV'].tolist()[0]  
        if self.policy_info['itc']>0:
            price_breakdown_ITC = price_breakdown.loc[price_breakdown['Name']=='One time capital incentive','NPV'].tolist()[0]
        else:
            price_breakdown_ITC = 0

        price_breakdown_renewables=price_breakdown_wind + price_breakdown_solar +price_breakdown_battery
        price_breakdown_total_h2_storage = price_breakdown_compression+price_breakdown_storage+price_breakdown_storage_FOM+price_breakdown_compressor_FOM
        price_breakdown_renewables_FOM = price_breakdown_wind_FOM + price_breakdown_solar_FOM + price_breakdown_battery_FOM
        
        lcoh_check = price_breakdown_electrolyzer+price_breakdown_electrolysis_FOM\
            + price_breakdown_desalination+price_breakdown_desalination_FOM+ price_breakdown_electrolysis_VOM\
                +price_breakdown_renewables+price_breakdown_renewables_FOM+price_breakdown_total_h2_storage+price_breakdown_taxes+price_breakdown_water+price_breakdown_grid_elec_price+remaining_financial\
                    -price_breakdown_ITC
        lcoh_breakdown = {'LCOH: Compressor ($/kg)':price_breakdown_compression,\
                        'LCOH: Hydrogen Storage ($/kg)':price_breakdown_storage,\
                        'LCOH: Hydrogen Storage FOM ($/kg)':price_breakdown_storage_FOM,\
                        'LCOH: Compressor FOM ($/kg)':price_breakdown_compressor_FOM,\
                        'LCOH: Electrolyzer CAPEX ($/kg)':price_breakdown_electrolyzer,'LCOH: Desalination CAPEX ($/kg)':price_breakdown_desalination,\
                        'LCOH: Electrolyzer FOM ($/kg)':price_breakdown_electrolysis_FOM,'LCOH: Electrolyzer VOM ($/kg)':price_breakdown_electrolysis_VOM,\
                        'LCOH: Desalination FOM ($/kg)':price_breakdown_desalination_FOM,\
                        'LCOH: Wind Plant ($/kg)':price_breakdown_wind,'LCOH: Wind Plant FOM ($/kg)':price_breakdown_wind_FOM,\
                        'LCOH: Solar Plant ($/kg)':price_breakdown_solar,'LCOH: Solar Plant FOM ($/kg)':price_breakdown_solar_FOM,\
                        'LCOH: Battery Storage ($/kg)':price_breakdown_battery,'LCOH: Battery Storage FOM ($/kg)':price_breakdown_battery_FOM,\
                        'LCOH: Taxes ($/kg)':price_breakdown_taxes,\
                        'LCOH: Water consumption ($/kg)':price_breakdown_water,'LCOH: Grid electricity ($/kg)':price_breakdown_grid_elec_price,\
                        'LCOH: ITC ($/kg)':-1*price_breakdown_ITC,\
                        'LCOH: Finances ($/kg)':remaining_financial,'LCOH: total ($/kg)':lcoh_check,'LCOH Profast:':sol['price']}
        return(sol,summary,price_breakdown,lcoh_breakdown)
    
    def get_lcoh2_storage_outputs(self,pf):
        sol = pf.solve_price()
    
        summary = pf.summary_vals
        
        price_breakdown = pf.get_cost_breakdown()
        total_price_capex =+ price_breakdown.loc[price_breakdown['Name']=='Compression','NPV'].tolist()[0]\
                      + price_breakdown.loc[price_breakdown['Name']=='Hydrogen Storage','NPV'].tolist()[0]
        capex_fraction = {'Compression':price_breakdown.loc[price_breakdown['Name']=='Compression','NPV'].tolist()[0]/total_price_capex,
                        'Hydrogen Storage':price_breakdown.loc[price_breakdown['Name']=='Hydrogen Storage','NPV'].tolist()[0]/total_price_capex}
        
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
        
        price_breakdown_compression = price_breakdown.loc[price_breakdown['Name']=='Compression','NPV'].tolist()[0] + cap_expense*capex_fraction['Compression']
        price_breakdown_storage = price_breakdown.loc[price_breakdown['Name']=='Hydrogen Storage','NPV'].tolist()[0]+cap_expense*capex_fraction['Hydrogen Storage']

        price_breakdown_storage_FOM = price_breakdown.loc[price_breakdown['Name']=='Hydrogen Storage Fixed O&M Cost','NPV'].tolist()[0]  
        price_breakdown_compressor_FOM = price_breakdown.loc[price_breakdown['Name']=='Compression Fixed O&M Cost','NPV'].tolist()[0]  
        price_breakdown_taxes = price_breakdown.loc[price_breakdown['Name']=='Income taxes payable','NPV'].tolist()[0]\
            - price_breakdown.loc[price_breakdown['Name'] == 'Monetized tax losses','NPV'].tolist()[0]\
                
        if self.gen_inflation > 0:
            price_breakdown_taxes = price_breakdown_taxes + price_breakdown.loc[price_breakdown['Name']=='Capital gains taxes payable','NPV'].tolist()[0]
        if self.policy_info['itc']>0:
            price_breakdown_ITC = price_breakdown.loc[price_breakdown['Name']=='One time capital incentive','NPV'].tolist()[0]
        else:
            price_breakdown_ITC = 0
        price_breakdown_total_h2_storage =price_breakdown_compression+price_breakdown_storage+price_breakdown_storage_FOM+price_breakdown_compressor_FOM

        lcoh_check = price_breakdown_total_h2_storage+price_breakdown_taxes+remaining_financial-price_breakdown_ITC

        lcoh_breakdown = {'LCOH: Compressor ($/kg)':price_breakdown_compression,\
                        'LCOH: Hydrogen Storage ($/kg)':price_breakdown_storage,\
                        'LCOH: Hydrogen Storage FOM ($/kg)':price_breakdown_storage_FOM,\
                        'LCOH: Compressor FOM ($/kg)':price_breakdown_compressor_FOM,\
                        'LCOH: Taxes ($/kg)':price_breakdown_taxes,\
                        'LCOH: ITC ($/kg)':-1*price_breakdown_ITC,\
                        'LCOH: Finances ($/kg)':remaining_financial,'LCOH: total ($/kg)':lcoh_check,'LCOH Profast:':sol['price']}
        return(sol,summary,price_breakdown,lcoh_breakdown)
    # capex_df_keys = ['Electrolysis system [$]','Compression [$]','Hydrogen Storage [$]','Desalination [$]','Wind Plant [$]','Solar Plant [$]','Battery [$]']
    # capex_df_vals=[price_breakdown.loc[price_breakdown['Name']=='Electrolysis system','NPV'].tolist()[0],\
    #                 price_breakdown.loc[price_breakdown['Name']=='Compression','NPV'].tolist()[0],\
    #                 price_breakdown.loc[price_breakdown['Name']=='Hydrogen Storage','NPV'].tolist()[0],\
    #                 price_breakdown.loc[price_breakdown['Name']=='Desalination','NPV'].tolist()[0],\
    #                 price_breakdown.loc[price_breakdown['Name']=='Wind Plant','NPV'].tolist()[0],\
    #                 price_breakdown.loc[price_breakdown['Name']=='Solar Plant','NPV'].tolist()[0],\
    #                 price_breakdown.loc[price_breakdown['Name']=='Battery Storage','NPV'].tolist()[0]]
    # capex_df = pd.Series(dict(zip(capex_df_keys,capex_df_vals)))
    # return(sol,lcoh_breakdown,capex_df)