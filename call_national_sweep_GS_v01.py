import os
import sys
#sys.path.append('')
from dotenv import load_dotenv
import pandas as pd
import yaml
from floris.utilities import Loader, load_yaml
# from hybrid.sites.site_info  import SiteInfo
# from tools.write_outputs_OCED import write_outputs #this needs to be completed
# from hybrid.sites.create_site_hpc import SiteInfo #this is not completed
# from tools.calc_distance_between_lat_lon #this is intended to calc distance to salt caverns
import time
import numpy as np
# from national_sweep_optimize_v01b import opt_national_sweep
from national_sweep_GS_v01 import opt_national_sweep
from analysis.GS_v01_set_PF_params import init_profast
import warnings
warnings.filterwarnings("ignore")
# import yaml
# class Loader(yaml.SafeLoader):

#     def __init__(self, stream):

#         self._root = os.path.split(stream.name)[0]

#         super().__init__(stream)

#     def include(self, node):

#         filename = os.path.join(self._root, self.construct_scalar(node))

#         with open(filename, 'r') as f:
#             return yaml.load(f, self.__class__)


# Loader.add_constructor('!include', Loader.include)

# def load_yaml(filename, loader=Loader):
#     if isinstance(filename, dict):
#         return filename  # filename already yaml dict
#     with open(filename) as fid:
#         return yaml.load(fid, loader)


# version = 'v01b'
# version = 'GS_v01'
parent_path = os.path.abspath('') + '/'
# run_desc = 'sweep' #or optimize

input_file_name = 'model_inputs.yaml'
filename = parent_path + 'input_info/' + input_file_name
config = load_yaml(filename)
run_desc = '{}_'.format(int(config['simulation']['re_cost_year_scenario'])) + config['optimization']['optimization_type']

pf_init = init_profast(config)
version =config['simulation']['desc']
num_sites_to_run = config['simulation']['n_sites']


# result_subdir = '{}_{}_{}k'.format(version,run_desc,int(num_sites_to_run/1000))
result_subdir = '{}_{}_{}sites'.format(version,run_desc,int(num_sites_to_run))
if run_desc=='simple_param':
    subsubdirectories = ['lcoh_results','sweep_results'] #do not modify for parametric sweep
else:
    #TODO: update this!
    subsubdirectories = ['lcoh_results','sweep_results'] 

#Specify results folder
#NOTE: Change results_main_directory if using HPC!
if len(config['outputs']['save_output_dir']) == 0:
    result_main_path= os.path.abspath('') + '/results/'
    # config['outputs']['save_output_dir'] = result_main_path
else:
    result_main_path= config['outputs']['save_output_dir']
results_main_directory = result_main_path + result_subdir + '/'
print('results will be saved to this location:')
print(results_main_directory)
config['outputs']['save_output_dir']= result_main_path + result_subdir + '/'
if not os.path.isdir(results_main_directory):
    os.makedirs(results_main_directory)
for subsubdir in subsubdirectories:
    if not os.path.isdir(results_main_directory + subsubdir + '/'):
        os.makedirs(results_main_directory + subsubdir + '/')

opt = opt_national_sweep(config)
sitelist_filename = parent_path + 'input_info/' + config['simulation']['site_list_filename']
site_list = pd.read_csv(sitelist_filename,index_col='Unnamed: 0')
if len(site_list)>num_sites_to_run:
    site_step = int(np.round(len(site_list)/num_sites_to_run))
    site_idx = np.arange(0,len(site_list),site_step)
else:
    site_idx = np.array(site_list.index.values)
for i,si in enumerate(site_idx):
    
    # site_desc = '{}_ID{}'.format(site_list.iloc[si]['state'].replace(' ',''),site_list.iloc[si]['site_id'])
    # site_desc = '{}_Num{}'.format(site_list.iloc[si]['state'].replace(' ',''),site_list.iloc[si].name)
    site_info = {'latitude':site_list.iloc[si]['latitude'],'longitude':site_list.iloc[si]['longitude'],'state':site_list.iloc[si]['state'].replace(' ','')}
    opt.run_site_outline(site_info,config)
    stop
    # opt.run_single_site(site_list.iloc[si],result_subdir)

# filename_atb = parent_path + 'input_info/' +'ATB2023_RECosts.yaml'
# filename_h2 = parent_path + 'input_info/' +'HydrogenStorage_Costs.yaml'
# filename_policy = parent_path + 'input_info/' +'Policy_Defaults.yaml'

# with open(filename_h2) as fid:
#     h2_config=yaml.load(fid,yaml.SafeLoader)
# fid.close()

# with open(filename_policy) as fid:
#     policy_config=yaml.load(fid,yaml.SafeLoader)
# fid.close()

# with open(filename_atb) as fid:
#     atb_config=yaml.load(fid,yaml.SafeLoader)
# fid.close()

# with open(filename) as fid:
#     config=yaml.load(fid,yaml.SafeLoader)
# fid.close()
# stop

# with open(filename) as fid:
#     config=yaml.load(fid,yaml.SafeLoader)
# fid.close()
# config2=load_yaml(filename)
# stopp
# #TODO: check yaml loading
# # return config
# # hub_height = 115 #we need 100m and 120m hub-height resource info




# results_main_directory = opt.main_dir + 'results/' + result_subdir + '/'
# if not os.path.isdir(results_main_directory):
#     os.makedirs(results_main_directory)
# for subsubdir in subsubdirectories:
#     if not os.path.isdir(results_main_directory + subsubdir + '/'):
#         os.makedirs(results_main_directory + subsubdir + '/')


# #initialize site_ID tracker
# site_id_tracker = np.zeros(len(site_idx))

# #Step 0: pre-initialize set-variables
# for i,si in enumerate(site_idx):
#     # lat=site_list.iloc[si]['latitude']
#     # lon=site_list.iloc[si]['longitude']
#     # approx_wind_cf = site_list.iloc[si]['capacity_factor']
#     site_id_tracker[i]=site_list.iloc[si]['site_id']
#     site_desc = '{}_ID{}'.format(site_list.iloc[si]['State'].replace(' ',''),site_list.iloc[si]['site_id'])
#     # site_list.iloc[si]
#     []
#     opt.run_single_site(site_list.iloc[si],result_subdir)
#     []
#     #if running multi-sites
#     # 
#     # 
#     #opt.run_all(site_list.iloc[start_idx:start_idx+step_size],result_subdir)
#     #start_idx +=step_size
    