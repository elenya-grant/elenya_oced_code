import yaml
import os
import sys
import numpy as np
import pandas as pd
# from floris_2_csv import csv_from_floris
# from csv_2_floristurb import csv_from_floris

class check_wind:
    def __init__(self,parent_path,turbine_size_mw):
        print('initalized check_wind')
        #floris_path = parent_path + 'floris_inputs_files/'
        #turb_lib = parent_path + 'hybrid/turbine_models/'
        self.turb_lib = parent_path + 'turbine_library/'
        current_files = os.listdir(self.turb_lib)
        self.turbine_size_mw = turbine_size_mw
        self.turb_files = [file for file in current_files if '{}MW'.format(turbine_size_mw) in file]
        # turbine_name = [turb_file.split('.')[0] for turb_file in self.turb_files if '.csv' in turb_file]
        # self.turbine_name = turbine_name[0]
        # if floris:
        #     self.check_for_floris_farm(wind_farm_capacity_mw,turbine_size_mw)
        # else:
        #     filename,rot_diam,hubht=self.check_for_pysam()
        
    def run(self,wind_farm_capacity_mw,floris):
        if floris:
            #print('using floris')
            output = self.check_for_floris_farm(wind_farm_capacity_mw)
        else:
            #print('using pysam')
            output =self.check_for_pysam()
        return output


    def check_for_floris_farm(self,wind_farm_capacity_mw):
        #print("check wind floris farm")
        from tools.windfarm_maker import make_windfarm
        
        floris_file_desc = [turb_file.split('.yaml')[0] for turb_file in self.turb_files if '.yaml' in turb_file]
        turbine_name = floris_file_desc[np.argmin([len(turb) for turb in floris_file_desc])]
        #min(len(turb) for turb in floris_file_desc )
        rot_diam,hubht = self.get_turb_specs(turbine_name)

        ref_farm_file = self.turb_lib + 'refFarm_' + turbine_name + '.yaml'
        wf=make_windfarm(ref_farm_file)
        nTurbs,actual_wind_farm_capacity = wf.calc_num_turbs(wind_farm_capacity_mw,self.turbine_size_mw)
        #farm_name = '{}turbs_{}MWfarm_{}'.format(round(wind_farm_capacity_mw/self.turbine_size_mw),int(wind_farm_capacity_mw),turbine_name)
        # farm_desc='{}MWfarm_{}'.format(int(wind_farm_capacity_mw),turbine_name)
        farm_desc='{}MWfarm_{}'.format(int(actual_wind_farm_capacity),turbine_name)
        
        farm_file = [file for file in floris_file_desc if farm_desc in file]
        if len(farm_file) > 0:
            #have the file
            floris_file= self.turb_lib + farm_file[0] + '.yaml'
            floris_config = self.load_floris_file(floris_file)
        else:
            "Making new floris farm layout..."
            #don't have the file
            new_farm_desc='{}turbs_{}MWfarm_{}'.format(nTurbs,actual_wind_farm_capacity,turbine_name)
            floris_file,floris_config=wf.make_new_farm(rot_diam,nTurbs,self.turb_lib,new_farm_desc)
            
        return {'filename':floris_file,'floris config':floris_config,'nTurbs':nTurbs,'Rotor Diameter':rot_diam,'Hub Height':hubht}
    def get_turb_specs(self,turbine_name):
        fl=self.read_floris_turb_file(turbine_name)
        rot_diam = fl['rotor_diameter']
        hubht=fl['hub_height']
        return rot_diam,hubht
    def load_floris_file(self,floris_file):
        from floris.utilities import Loader, load_yaml
        floris_config=load_yaml(floris_file)
        return floris_config
    def check_for_pysam(self):
        print("checking for pysam")
        turbine_name = [turb_file.split('.csv')[0] for turb_file in self.turb_files if '.csv' in turb_file]
        if len(turbine_name)>0:
            turbine_name = turbine_name[0]
            csv_filename= self.turb_lib + turbine_name + '.csv'
            rot_diam,hubht = self.get_turb_specs(turbine_name)
            #floris_file_desc = [turb_file.split('.')[0] for turb_file in self.turb_files if '.yaml' in turb_file]
            #turbine_name = min(len(turb) for turb in floris_file_desc )
            # fl=self.read_floris_turb_file(turbine_name)
            # rot_diam = fl['rotor_diameter']
            # hubht=fl['hub_height']
        else:
            csv_filename = None
            rot_diam = None
            hubht= None
            print("Can't find CSV file for files that use {} turbine".format(self.turb_files))
        return {'filename':csv_filename,'Rotor Diameter':rot_diam,'Hub Height':hubht}
        #return csv_filename,rot_diam,hubht

    def read_floris_turb_file(self,turbine_name):
        filename = self.turb_lib + turbine_name + '.yaml'
        with open(filename) as fid:
            config=yaml.load(fid,yaml.SafeLoader)
        fid.close()
        return config






        
        pass
if __name__=="__main__":
    parent_path = '/Users/egrant/Desktop/modular_hopp/green_heart/'
    checker = check_wind(parent_path,6,2,True)
    turbine_name = 'SGRE_6MW'
    flturb=checker.read_floris_turb_file(turbine_name)
    []
