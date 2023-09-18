
import numpy as np
import pandas as pd

import sys

def init_profast(config):
    pf_dir = config['simulation']['profast_folder']
    sys.path.insert(1,pf_dir)
    import ProFAST
    gen_inflation=config["finance_parameters"]["general_inflation"]
    # config["finance_parameters"][""]
    land_cost=config["finance_parameters"]["land_cost"]
    plant_life = config["plant_general"]["plant_life"]
    pf = ProFAST.ProFAST('blank')
    pf.set_params('commodity',{"name":'Hydrogen',"unit":"kg","initial price":100,"escalation":gen_inflation})
    # pf.set_params('capacity',electrolysis_plant_capacity_kgperday) #units/day
    pf.set_params('maintenance',{"value":0,"escalation":gen_inflation})
    pf.set_params('analysis start year',config["plant_general"]["start_year"])
    pf.set_params('operating life',plant_life)
    pf.set_params('installation months',config["plant_general"]["installation_months"])
    pf.set_params('installation cost',{"value":0,"depr type":"Straight line","depr period":4,"depreciable":False})
    pf.set_params('non depr assets',land_cost)
    pf.set_params('end of proj sale non depr assets',land_cost*(1+gen_inflation)**plant_life)
    pf.set_params('demand rampup',0)
    # pf.set_params('long term utilization',elec_cf) #TODO
    pf.set_params('credit card fees',0)
    pf.set_params('sales tax',config["finance_parameters"]["sales_tax_rate"]) 
    pf.set_params('license and permit',{'value':00,'escalation':gen_inflation})
    pf.set_params('rent',{'value':0,'escalation':gen_inflation})
    pf.set_params('property tax and insurance percent',config["finance_parameters"]["property_tax"] + config["finance_parameters"]["property_insurance"])
    # pf.set_params('property tax and insurance',config["finance_parameters"]["property_tax"] + config["finance_parameters"]["property_insurance"]) #from updated PF
    pf.set_params('admin expense percent',0)
    # pf.set_params('admin expense',0) #PF update
    pf.set_params('total income tax rate',config["finance_parameters"]["total_income_tax_rate"])
    pf.set_params('capital gains tax rate',config["finance_parameters"]["capital_gains_tax_rate"])
    pf.set_params('sell undepreciated cap',True)
    pf.set_params('tax losses monetized',True)
    pf.set_params('general inflation rate',gen_inflation)
    pf.set_params('leverage after tax nominal discount rate',config["finance_parameters"]["discount_rate"])
    pf.set_params('debt equity ratio of initial financing',config["finance_parameters"]["debt_equity_split"]/(100-config["finance_parameters"]["debt_equity_split"])) #TODO
    pf.set_params('debt type','Revolving debt')
    pf.set_params('debt interest rate',config["finance_parameters"]["debt_interest_rate"]) #MAYBE TODO
    pf.set_params('cash onhand percent',config["finance_parameters"]["cash_onhand_months"])
    # pf.set_params('cash onhand',config["finance_parameters"]["cash_onhand_months"]) #PF
    return pf