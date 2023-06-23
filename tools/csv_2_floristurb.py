import yaml
import os
import sys
import pandas as pd
import numpy as np
class floris_from_csv:
    def __init__(self,parent_path,turbine_name):
        #floris_path = parent_path + 'floris_inputs_files/'
        self.turb_dir = parent_path + 'turbine_library/'

        if 'VestasV82' in turbine_name:
            self.rotor_diameter = 82
            self.turbine_rating_mw = 1.65
            self.tower_height = 80
            self.tsr=8.737
        elif 'SGRE_6MW' in turbine_name:
            self.rotor_diameter = 170
            self.turbine_rating_mw = 6
            self.tower_height = 115
            self.tsr=7.9
    def load_reference_floris_turb(self):
        ref_config_filename = 'turbine_reference.yaml'
        filename = self.turb_dir + ref_config_filename
        with open(filename) as fid:
            config=yaml.load(fid,yaml.SafeLoader)
        fid.close()
        return config
    def load_reference_floris_farm(self):
        from floris.utilities import Loader, load_yaml
        ref_config_filename = 'farm_reference.yaml'
        filename = self.turb_dir + ref_config_filename
        farm_ref=load_yaml(filename)
        # with open(filename) as fid:
        #     config=yaml.load(fid,yaml.SafeLoader)
        # fid.close()
        return farm_ref


    def load_csv(self,turbine_name):
        power_df=pd.read_csv(self.turb_dir + turbine_name + '.csv',index_col='Wind Speed [m/s]')
        #power_df=pd.read_csv(self.turb_dir + self.custom_powercurve_path,index_col='Wind Speed [m/s]')
        idx=np.argwhere(np.logical_not(np.isnan(power_df.index.values)))[:,0]
        cp=list(power_df['Cp [-]'].values[idx])
        ct=list(power_df['Ct [-]'].values[idx])
        v=list(power_df.index.values[idx])
        return cp,ct,v
    def write_yaml(self,turbine_name,config):
        filename = self.turb_dir + turbine_name + '.yaml'
        with open(filename, "w+") as f:
            yaml.dump(config,f,sort_keys=False,default_flow_style=False)
            # ryaml.dump(config,f) #,sort_keys=False,default_flow_style=False)
        print("Saved {} to {}".format(turbine_name,self.turb_dir))

    def make_floris_turbine_file(self,turbine_name):
        cp,ct,v=self.load_csv(turbine_name)
        ref=self.load_reference_floris_turb()
        ref['power_thrust_table']['power']=[float(i) for i in cp]
        ref['power_thrust_table']['thrust']=[float(i) for i in ct]
        ref['power_thrust_table']['wind_speed']=[float(i) for i in v]
        turbine_type = {'turbine_type':turbine_name,'TSR':self.tsr,'hub_height':self.tower_height,'rotor_diameter':self.rotor_diameter}
        ref.update(turbine_type)

        self.write_yaml(turbine_name,ref)

    def make_floris_farm_file(self,turbine_name):
        ref=self.load_reference_floris_farm()
        self.ref['flow_field']['reference_wind_height']=self.tower_height
        #self.ref_config['farm']['turbine_type'][0]



    
    
