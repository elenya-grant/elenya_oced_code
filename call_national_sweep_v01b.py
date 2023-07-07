import os
import sys
#sys.path.append('')
from dotenv import load_dotenv
import pandas as pd
# from hybrid.sites.site_info  import SiteInfo
# from tools.write_outputs_OCED import write_outputs #this needs to be completed
# from hybrid.sites.create_site_hpc import SiteInfo #this is not completed
# from tools.calc_distance_between_lat_lon #this is intended to calc distance to salt caverns
import time
import numpy as np
from national_sweep_optimize_v01b import opt_national_sweep
import warnings
warnings.filterwarnings("ignore")
# import yaml


version = 'v01b'
parent_path = os.path.abspath('') + '/'
num_sites_to_run = 50000
run_desc = 'sweep' #or optimize

#Specify results folder
# result_directory = parent_path + 'results/{}_{}_{}k/'.format(version,run_desc,int(num_sites_to_run/1000))
result_subdir = '{}_{}_{}k'.format(version,run_desc,int(num_sites_to_run/1000))
subsubdirectories = ['lcoh_results','sweep_results'] #do not modify for parametric sweep
default_input_file = parent_path + 'input_info/sweep_defaults.csv'
model_params = pd.read_csv(default_input_file,index_col = 'Variable')

# hub_height = 115 #we need 100m and 120m hub-height resource info
sitelist_100k_filename = parent_path + 'input_info/wtk_site_metadata_100k_sites.csv'
site_list = pd.read_csv(sitelist_100k_filename,index_col='Unnamed: 0')
site_step = int(np.round(len(site_list)/num_sites_to_run))
site_idx = np.arange(0,len(site_list),site_step)

opt = opt_national_sweep(model_params)

results_main_directory = opt.main_dir + 'results/' + result_subdir + '/'
if not os.path.isdir(results_main_directory):
    os.makedirs(results_main_directory)
for subsubdir in subsubdirectories:
    if not os.path.isdir(results_main_directory + subsubdir + '/'):
        os.makedirs(results_main_directory + subsubdir + '/')


#initialize site_ID tracker
site_id_tracker = np.zeros(len(site_idx))

#Step 0: pre-initialize set-variables
for i,si in enumerate(site_idx):
    # lat=site_list.iloc[si]['latitude']
    # lon=site_list.iloc[si]['longitude']
    # approx_wind_cf = site_list.iloc[si]['capacity_factor']
    site_id_tracker[i]=site_list.iloc[si]['site_id']
    site_desc = '{}_ID{}'.format(site_list.iloc[si]['State'].replace(' ',''),site_list.iloc[si]['site_id'])
    # site_list.iloc[si]
    []
    opt.run_single_site(site_list.iloc[si],result_subdir)
    []
    #if running multi-sites
    # 
    # 
    #opt.run_all(site_list.iloc[start_idx:start_idx+step_size],result_subdir)
    #start_idx +=step_size
    