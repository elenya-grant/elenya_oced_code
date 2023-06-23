from analysis.hopp_for_h2 import hopp_for_h2
from analysis.simple_dispatch import SimpleDispatch
from analysis.hopp_for_h2_floris import hopp_for_h2_floris
import numpy as np
import yaml
class run_hybrid_plant:
    def __init__(self,run_wind,run_solar,floris):
        if run_wind:
            self.run_wind_plant = True
        else:
            self.run_wind_plant = False
        if run_solar:
            self.run_solar_plant = True
        else:
            self.run_solar_plant = False
        self.floris = floris
    

    def update_components_to_run(self,site,run_wind,run_solar):
        if run_wind:
            site.data['no_wind'] == False
            self.run_wind_plant = True
            
        else:
            site.data['no_wind'] == True
            self.run_wind_plant = False
        if run_solar:
            site.data['no_solar'] == False
            self.run_solar_plant = True
        else:
            site.data['no_solar'] == True
            self.run_solar_plant = False
        return site
    def update_site(self,resource_year,hub_height):
        #note - should move this somewhere else 
        #will need general site info as input, not the object
        #then will need to return the created object
        pass
    def update_scenario(self,stuff):
        pass
    def update_technologies(self,stuff):
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
    def run_wind_solar_generation(self,site,scenario,technologies):
        #NOTE: I should make tools to update specific things depending on optimizations
        #ex: if you're just looking at pem sizing, don't need to re-run plant

        #also add tool to see if we have an existing power generation profile for a specific scenario
        #so we wouldn't have to rerun hybrid plant
        hybrid_plant=self.make_hybrid_plant(technologies,site,scenario)
        wind_farm_generation = self.get_wind_generation(hybrid_plant)
        solar_farm_generation = self.get_solar_generation(hybrid_plant)

        wind_solar_generation = wind_farm_generation + solar_farm_generation
        return wind_solar_generation

    def run_power_generations(self,RE_info,site,scenario,load_kWh):
        technologies = self.make_technologies(RE_info)
        hybrid_plant = self.run_wind_solar_plant(technologies,site,scenario)
        wind_pv_power_kW,wind_pv_breakout = self.process_hybrid_plant(hybrid_plant)
        wind_pv_shortfall, wind_pv_curtailment = self.calc_shortfall_curtailment(load_kWh,wind_pv_power_kW)
        battery_dispatched = self.run_battery(RE_info['battery'],wind_pv_shortfall,wind_pv_curtailment )
        #TODO: add in battery to run for multiple batteries
        wind_pv_bat_power =  wind_pv_power_kW + battery_dispatched - wind_pv_curtailment 
        
        wind_pv_bat_shortfall, wind_pv_bat_curtailment = self.calc_shortfall_curtailment(load_kWh,wind_pv_bat_power)

        renewable_plant_generation = {
        'Load Demand':load_kWh,
        'Wind + PV Generation':wind_pv_power_kW,
        'Wind + PV Shortfall':wind_pv_shortfall,
        'Wind + PV Curtailment':wind_pv_curtailment,
        'Battery Used':battery_dispatched,
        'Wind + PV + Battery':wind_pv_bat_power,
        'Wind + PV + Battery Shortfall':wind_pv_bat_shortfall,
        'Wind + PV + Battery Curtailment':wind_pv_bat_curtailment}
        renewable_plant_generation.update(wind_pv_breakout)
        return renewable_plant_generation

    def init_solar(self,technologies,solar_size_mw):
        
        if solar_size_mw>0:
                technologies['pv']={'system_capacity_kw': solar_size_mw * 1000}
        
        return technologies
    def init_pysam_wind(self,technologies,wind_info):
        #{'filename':csv_filename,'Rotor Diameter':rot_diam,'Hub Height':hubht}
        #need scenario to have 'Powercurve File'
        technologies['wind']={'num_turbines': np.floor(wind_info['wind_farm_capacity']/ wind_info['turbine_rating_mw']),
                                        'turbine_rating_kw': wind_info['turbine_rating_mw']*1000,
                                        'hub_height': wind_info['Hub Height'],
                                        'rotor_diameter': wind_info['Rotor Diameter']}

        #TODO: add 'Powercurve File' to scenario
        return technologies
    def init_floris(self,technologies,wind_info):
        
        
        technologies['wind']= {'num_turbines':wind_info['nTurbs'],
                                    'turbine_rating_kw': wind_info['turbine_rating_mw']*1000,
                                    'model_name': 'floris',
                                    'timestep': [0,8760],
                                    'floris_config': wind_info['floris config']}
        
        return technologies
    def make_technologies(self,RE_info):
        technologies = {}
        if self.run_wind_plant:
            if self.floris:
                technologies = self.init_floris(technologies,RE_info['wind'])

            else:
                technologies = self.init_pysam_wind(technologies,RE_info['wind'])
        if self.run_solar_plant:
            technologies =self.init_solar(technologies,RE_info['solar']['solar_size_mw'])

        return technologies



    # def run_wind_solar_plant(self,technologies,site,scenario):
    def make_hybrid_plant(self,technologies,site,scenario):
        storage_size_mw_trash = 0
        storage_size_mwh_trash = 0
        storage_hours_trash = 0
        sim_years = site.n_timesteps/8760
        if self.run_solar_plant:
            solar_size_mw=technologies['pv']['system_capacity_kw']/1000
        else:
            solar_size_mw = 0

        # param load: ``list``,
        # (8760) hourly load profile of electrolyzer in kW. Default is continuous load at kw_continuous rating
        #NOTE: numbers below should be unused and are just temporary
        load = np.zeros(8760)
        interconnection_size_mw = 50
        kw_continuous = interconnection_size_mw*1000
        if self.run_wind_plant:
            wind_size_mw = (technologies['wind']['num_turbines']*technologies['wind']['turbine_rating_kw'])/1000
        else:
            wind_size_mw = 0
        if self.floris:
            custom_powercurve=False
            grid_connected_hopp = False #why
            hybrid_plant=hopp_for_h2_floris(site, scenario, technologies,
                            wind_size_mw, solar_size_mw, storage_size_mw_trash, storage_size_mwh_trash, storage_hours_trash,
                            kw_continuous, load,
                            custom_powercurve,
                            interconnection_size_mw, grid_connected_hopp,sim_years)
        else:
            grid_connected_hopp = True #why
            custom_powercurve=True
            hybrid_plant=hopp_for_h2(site, scenario, technologies,
                            wind_size_mw, solar_size_mw, storage_size_mw_trash, storage_size_mwh_trash, storage_hours_trash,
                            kw_continuous, load,
                            custom_powercurve,
                            interconnection_size_mw, grid_connected_hopp,sim_years)
        return hybrid_plant
    def get_wind_generation(self,hybrid_plant):
        plant_power = np.zeros(8760)
        if self.run_wind_plant:
            wind_plant_power = np.array(hybrid_plant.wind.generation_profile[0:len(plant_power)])#[0:8759]
        else:
            wind_plant_power = np.zeros(len(plant_power)) #kWh per hour
        return wind_plant_power 
    def get_solar_generation(self,hybrid_plant):
        plant_power = np.zeros(8760)
        solar_plant_power = np.array(hybrid_plant.pv.generation_profile[0:len(plant_power)])
        if self.run_solar_plant:
            solar_plant_power = np.array(hybrid_plant.pv.generation_profile[0:len(plant_power)])
        else:
            solar_plant_power = np.zeros(len(plant_power))#kWh per hour
        return solar_plant_power

    def process_hybrid_plant(self,hybrid_plant):
        plant_power = np.zeros(8760)
        if self.run_wind_plant:
            wind_plant_power = np.array(hybrid_plant.wind.generation_profile[0:len(plant_power)])#[0:8759]
        else:
            wind_plant_power = np.zeros(len(plant_power))

        if self.run_solar_plant:
            solar_plant_power = np.array(hybrid_plant.pv.generation_profile[0:len(plant_power)])
        else:
            solar_plant_power = np.zeros(len(plant_power))
        combined_pv_wind_power_production_hopp = np.array(solar_plant_power) + np.array(wind_plant_power)
        split_power_profile={'wind_power_kWh':wind_plant_power,'solar_power_kWh':solar_plant_power}
        return combined_pv_wind_power_production_hopp,split_power_profile

    # def calc_shortfall_curtailment(self,load_kW,power_kW):
    #     diff = np.array(power_kW) - np.array(load_kW)
    #     shortfall = np.where(diff<0, -1*diff, 0)
    #     curtailment =  np.where(diff>0, diff, 0)
    #     return shortfall, curtailment
    def calc_shortfall(self,shortfall_setpoint,actual_production):
        if isinstance(shortfall_setpoint,(float,int)):
            shortfall_setpoint=shortfall_setpoint*np.ones(len(actual_production))
        diff = np.array(shortfall_setpoint) - np.array(actual_production)
        shortfall = np.where(diff>0, diff, 0)
        return shortfall
    def calc_curtailment(self,curtailment_setpoint,actual_production):
        if isinstance(curtailment_setpoint,(float,int)):
            curtailment_setpoint=curtailment_setpoint*np.ones(len(actual_production))
        diff = np.array(actual_production) - np.array(curtailment_setpoint) 
        shortfall = np.where(diff>0, diff, 0)
        return shortfall
    def run_simple_battery(self,bat_info,shortfall_kW,curtailment_kW):
        # n_bat = bat_info['num_batteries']
        #TODO: add dispatch to multiple batteries
        
        bat_model = SimpleDispatch()
        bat_model.Nt = len(shortfall_kW)
        bat_model.curtailment = curtailment_kW
        bat_model.shortfall = shortfall_kW
        bat_model.charge_rate = bat_info['battery_size_mw']*1000
        bat_model.discharge_rate = bat_info['battery_size_mw']*1000
        bat_model.battery_storage = bat_info['battery_size_mwh']*1000

        battery_used, excess_energy, battery_SOC = bat_model.run()
        return np.array(battery_used)
    def general_simple_dispatch(self,storage_capacity,storage_ramp_rate,shortfall,curtailment,dispatch_losses=0):
        
        h2_dispatch = SimpleDispatch()
        h2_dispatch.Nt = len(shortfall)
        h2_dispatch.curtailment = curtailment #array of curtailed units
        h2_dispatch.shortfall = shortfall #array of unit shortfall
        h2_dispatch.charge_rate = storage_ramp_rate #unit/hour
        h2_dispatch.discharge_rate = storage_ramp_rate #unit/hour
        h2_dispatch.battery_storage = storage_capacity #unit-hour

        unit_dispatched, excess_units, storage_SOC = h2_dispatch.run()
        unit_dispatched = unit_dispatched*((100-dispatch_losses)/100)
        #    combined_pv_wind_storage_power_production_hopp = combined_pv_wind_power_production_hopp + battery_used - combined_pv_wind_curtailment_hopp
        return np.array(unit_dispatched),np.max(storage_SOC)
    def hydrogen_simple_storage_dispatch(self,h2_storage_info,shortfall_kg,curtailment_kg):
        #n_bat = bat_info['num_batteries']
        #TODO: add dispatch to multiple batteries
        
        h2_dispatch = SimpleDispatch()
        h2_dispatch.Nt = len(shortfall_kg)
        h2_dispatch.curtailment = curtailment_kg
        h2_dispatch.shortfall = shortfall_kg
        h2_dispatch.charge_rate = h2_storage_info['storage_charge_rate_kg'] #kg 
        h2_dispatch.discharge_rate = h2_storage_info['storage_charge_rate_kg']#kg 
        h2_dispatch.battery_storage = h2_storage_info['storage_capacity_kg']#kg-hour

        kgh2_dispatched, excess_h2, h2_SOC_kg = h2_dispatch.run()
        #    combined_pv_wind_storage_power_production_hopp = combined_pv_wind_power_production_hopp + battery_used - combined_pv_wind_curtailment_hopp
        return np.array(kgh2_dispatched)
        
    
