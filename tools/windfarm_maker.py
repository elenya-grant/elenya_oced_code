import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt 
import os
import sys
from floris.utilities import Loader, load_yaml
#from tools.yaml_helper import Loader,load_yaml
# lat=46.716242
# lon=-122.954313
import yaml


#file_desc = 'Y{}_Wind{}_{}Turbs_{}TurbSize_Solar{}_Battery{}MW_{}hr_{}_{}_PEMCap{}MW_{}_{}'.format(atb_year,wind_size_mw,nTurbs,turbine_rating_mw,solar_size_mw,storage_size_mw,storage_hours,policy_option,grid_connection_scenario,electrolyzer_size_mw,windmodel_string,storage_type.replace(' ','_') )
#elenya_path = '/Users/egrant/Desktop/HOPP-GIT/Centralia_WA/'
class make_windfarm:
    def __init__(self,floris_refFarm_file):
        self.fl=load_yaml(floris_refFarm_file)

        

    def make_new_farm(self,rotor_diameter,nTurbs,save_yaml_dir,save_desc):
        new_layout=self.make_farm_layout(rotor_diameter,nTurbs)
        
        self.fl['farm'].update(new_layout)
        floris_filename = self.write_yaml(self.fl,save_yaml_dir,save_desc)

        return floris_filename,self.fl
    def calc_num_turbs(self,wind_farm_capacity_mw,turbine_rating_mw):
        nTurbs_init = np.ceil(wind_farm_capacity_mw/turbine_rating_mw)
        divisible_by_vals = np.arange(2,16,1)
        idx_divisble_by = np.where((nTurbs_init % divisible_by_vals) == 0)[0]
        if len(idx_divisble_by)==0:
            #remainder = nTurbs_init % divisible_by_vals
            quotient = (nTurbs_init // divisible_by_vals) + 1
            temp_nTurbs = quotient*divisible_by_vals
            capac_diff =  np.abs((temp_nTurbs*turbine_rating_mw) -wind_farm_capacity_mw)
            i=np.argmin(capac_diff)
            nTurbs=int(temp_nTurbs[i])
            # idx_not_divisible_by = np.where((nTurbs % divisible_by_vals) == 0)[0]
        else:
            nTurbs = int(nTurbs_init)
        
        actual_wind_farm_capacity = turbine_rating_mw*nTurbs
        return nTurbs,actual_wind_farm_capacity
    def update_turbine_file(self):
        #TODO: check if already exists
        ref = self.reference_turb_file()
        turbine_type = {'turbine_type':self.turbine_type,'TSR':self.tsr,'hub_height':self.tower_height,'rotor_diameter':self.rotor_diameter}
        ref.update(turbine_type)

        power_df=pd.read_csv(self.turb_dir + self.custom_powercurve_path,index_col='Wind Speed [m/s]')
        idx=np.argwhere(np.logical_not(np.isnan(power_df.index.values)))[:,0]
        cp=list(power_df['Cp [-]'].values[idx])
        ct=list(power_df['Ct [-]'].values[idx])
        v=list(power_df.index.values[idx])
        #floris_float_type = np.float64
        #power_table = {'power':list(power_df['Cp [-]'].values[idx]),'thrust':list(power_df['Ct [-]'].values[idx]),'wind_speed':list(power_df.index.values[idx])}
        ref['power_thrust_table']['power']=[float(i) for i in cp]#list(np.array(cp,dtype=np.float64))
        ref['power_thrust_table']['thrust']=[float(i) for i in ct]#list(np.array(ct,dtype=np.float64))
        ref['power_thrust_table']['wind_speed']=[float(i) for i in v]#list(np.array(v,dtype=np.float64))
        #self.save_turbine_file(ref)
        #self.write_yaml(ref,'tabletest' + self.turbine_type)
        return ref
    # def save_turbine_file(self,ref):
    #     import ruamel.yaml
    #     yaml= ruamel.yaml.YAML()
    #     #yaml.version = (1,2)
    #     yaml.default_flow_style=False
        
        
    #     #ruamel.yaml.dump(ref['power_thrust_table'],sys.stdout)
    #     yaml.dump(ref,sys.stdout)
    #     #yaml.dump(ref,f) 
    #     self.turb_dir
    def update_floris_main_file(self):
        ref_config = self.read_reference_yaml()
        ref_config['flow_field']['reference_wind_height']=self.tower_height
        nTurbs,new_layout = self.make_farm_layout()
        ref_config['farm'].update(new_layout)
        turb_type=self.update_turbine_file()
        ref_config['farm']['turbine_type'][0].update(turb_type)
        
        self.write_yaml(ref_config,'{}TurbFarm_{}'.format(int(nTurbs),self.turbine_type))
        []



    def make_turbine_type(self):
        #['farm']['turbine_type']
        #['flow_field']['reference_wind_height']
        #vars_to_overwrite = ['TSR','hub_height','rotor_diameter','turbine_type']
        turbine_type = {'TSR':self.tsr,'hub_height':self.tower_height,'rotor_diameter':self.rotor_diameter}
        self.ref_config['farm']['turbine_type'][0].update(turbine_type)
        self.ref_config['flow_field']['reference_wind_height']=self.tower_height
        print("updated turbine type and flow field")
        #self.ref_config.update({'keys_to_replace':vals_to_replace})
    def make_power_table(self):
        #['farm']['turbine_type']['power_thrust_table']
        power_df=pd.read_csv(self.turb_dir + self.custom_powercurve_path,index_col='Wind Speed [m/s]')
        idx=np.argwhere(np.logical_not(np.isnan(power_df.index.values)))[:,0]
        power_table = {'power':list(power_df['Cp [-]'].values[idx]),'thrust':list(power_df['Ct [-]'].values[idx]),'wind_speed':list(power_df.index.values[idx])}
        
        self.ref_config['farm']['turbine_type'][0]['power_thrust_table'].update(power_table)
        print("updated power thrust table")
    
    def make_farm_layout(self,rotor_diameter,nTurbs,d_spacing=5):
        D_apart = d_spacing*rotor_diameter #4? how many rotor diameters to space turbines
        
        possible_turbs_x = np.arange(2,nTurbs+1,1)
        possible_turbs_y = nTurbs/possible_turbs_x
        idx_possible_configs=np.argwhere(np.round(possible_turbs_y) - possible_turbs_y ==0)[:,0]
        nturbs_y = possible_turbs_y[idx_possible_configs]
        nturbs_x = possible_turbs_x[idx_possible_configs]
        diff_x_y = np.abs(nturbs_x-nturbs_y)
        most_square_indx = np.argmin(diff_x_y)
        nx=nturbs_x[most_square_indx]
        ny=nturbs_y[most_square_indx]

        x_pos = np.arange(0,nx*D_apart,D_apart)
        y_pos = np.arange(0,ny*D_apart,D_apart)

        layout_y = np.repeat(y_pos,int(nx))
        layout_x = np.tile(x_pos,int(ny))
        if len(layout_x) != nTurbs:
            print('issue in layout')
        else:
            layout_x_final = [float(x) for x in layout_x]
            layout_y_final = [float(y) for y in layout_y]
            # new_layout = {'layout_x':list(layout_x),'layout_y':list(layout_y)}
            new_layout = {'layout_x':layout_x_final,'layout_y':layout_y_final}
            #self.ref_config['farm'].update(new_layout)
            #wind_actual_capacity_mw = nTurbs*turbine_rating_mw
            #self.wind_actual_capacity_mw = nTurbs*self.turbine_rating_mw
            #print('updated wind farm layout, actual capacity is {} MW'.format(self.wind_actual_capacity_mw))
        return new_layout #,wind_actual_capacity_mw
    def make_new_floris_config(self,d_spacing =5):
        self.make_turbine_type()
        self.make_power_table()
        nTurbs=self.make_farm_layout(d_spacing)
        
        return self.ref_config,nTurbs
        #['farm']['layout_x'] #inc
        # ['farm']['layout_y'] #
    def read_reference_yaml(self):
        
        #Loader = yaml.SafeLoader
        #https://github.com/NREL/floris/blob/main/floris/utilities.py
        #https://github.com/NREL/floris/blob/main/floris/utilities.py
        # save_yaml_dir = '/Users/egrant/Desktop/modular_hopp/green_heart/hybrid/turbine_models/'
        # ref_config_filename = 'floris_input_Site1_offgrid.yaml'
        save_yaml_dir = '/Users/egrant/Desktop/modular_hopp/green_heart/floris_input_files/'
        ref_config_filename = 'floris_input4MW_lbw.yaml'
        
        
        filename = save_yaml_dir + ref_config_filename
        fl=load_yaml(filename)
        []
        with open(filename) as fid:
            config=yaml.load(fid,yaml.SafeLoader)
        fid.close()
        return config
    def reference_turb_file(self):
        
        turb_dir = '/Users/egrant/Desktop/modular_hopp/green_heart/hybrid/turbine_models/'
        ref_config_filename = 'lbw_4MW.yaml'
        filename = turb_dir + ref_config_filename
        with open(filename) as fid:
            config=yaml.load(fid,yaml.SafeLoader)
        fid.close()
        return config
        


    def write_yaml(self,config,save_yaml_dir,desc):
        #ryaml= ruamel.yaml.YAML()
        
        # ryaml.version = (1,2)
        #save_yaml_dir = '/Users/egrant/Desktop/modular_hopp/green_heart/hybrid/turbine_models/'
        file_desc = desc + '.yaml'
        filename = save_yaml_dir + file_desc
        with open(filename, "w+") as f:
            yaml.dump(config,f,sort_keys=False,default_flow_style=False)
            # ryaml.dump(config,f) #,sort_keys=False,default_flow_style=False)
        print("Saved {} to {}".format(file_desc,save_yaml_dir))
        return filename


    #file=open("person_data.yaml","w")
    #yaml.dump(dict,file)
    #file.close()
    # hub_height 
    #784
    
    
    # power_table_dict = {'turbine_type':turbine_type,'generator_efficiency':1.0,
    # 'hub_height':tower_height,'pP':1.88,'pT':1.88,'rotor_diameter':rotor_diameter,
    # 'TSR':tsr_op,'ref_density_cp_ct':1.225,
    # 'power_thrust_table':{'power':Cp,'thrust':Ct,'wind_speed':ws}}
    # farm_dict = {'layout_x':turb_x_pos,'layout_y':turb_y_pos,'turbine_type':{}}
    # floris_dict = {'description':'Three turbines using Gauss model',
    # 'farm': farm_dict}
if __name__=="__main__":
    #turb_size_mw = 1.65
    #turb_name = 'VestasV82_1.65MW_82'
    # ideal_nturbs = 196
    turb_name ='6MW_Centralia'
    
    turb_size_mw = 6
    ideal_nturbs = 32
    ideal_wind_cap = ideal_nturbs*turb_size_mw
    small_turbs = elenya_make_windfarm(ideal_wind_cap,turb_name)

    fi=small_turbs.read_reference_yaml()
    ref_turb = small_turbs.reference_turb_file()
    small_turbs.write_yaml(fi)
    []

    #small_turbs.estimate_tsr(13,10.8,70,41)
    save_yaml_dir = '/Users/egrant/Desktop/modular_hopp/green_heart/hybrid/turbine_models/'
    yaml_template = save_yaml_dir + "test6MW_Centralia.yaml"# + 'floris_input_Site1_offgrid.yaml'
    new_config,nTurbs=small_turbs.make_new_floris_config()
    small_turbs.write_yaml(new_config)
    #ruamel.yaml.load()
    ref_config_filename = 'floris_input_Site1_offgrid'
    #https://github.com/NREL/floris/blob/main/floris/simulation/floris.py
    file=open(yaml_template,"w")
    ruamel.yaml.dump(new_config,file,Dumper = ruamel.yaml.RoundTripDumper)
    # file=open(yaml_template,"w")
    # yaml.dump(new_config,file)
    file.close()
    []

