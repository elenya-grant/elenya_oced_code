import yaml
import os
import sys
import numpy as np
import pandas as pd
class csv_from_floris:
    def __init__(self,parent_path,turbine_name):
        #floris_path = parent_path + 'floris_inputs_files/'
        turb_lib = parent_path + 'turbine_library/'
        # turb_lib = parent_path
        filename =turb_lib + turbine_name + '.yaml'
        self.run(filename)
        #keys = ['Wind Speed [m/s]','Cp [-]','Ct [-]']
    def load_floris_turb(self,filename):
        with open(filename) as fid:
            config=yaml.load(fid,yaml.SafeLoader)
        fid.close()
        return config
    def run(self,filename):
        keys = ['Wind Speed [m/s]','Power [kW]','Cp [-]','Ct [-]']

        fi=self.load_floris_turb(filename)
        rho=fi['ref_density_cp_ct']
        area = np.pi*((fi['rotor_diameter']/2)**2)
        v=np.array(fi['power_thrust_table']['wind_speed'])
        wind_power_kW = 0.5*rho*area*(v**3)/1000
        cp=np.array(fi['power_thrust_table']['power'])
        ct=np.array(fi['power_thrust_table']['thrust'])
        power_kW = cp*wind_power_kW
        vals = [v,power_kW,cp,ct]


        cpctdf=pd.DataFrame(dict(zip(keys,vals)))
        csv_filename = filename.replace('.yaml','.csv')
        cpctdf.to_csv(csv_filename)

    # def check_for_pysam(self):
    #     file_type = '.csv'
    #     keys = ['Wind Speed [m/s]','Power [kW]','Cp [-]','Ct [-]']
    #     pass
