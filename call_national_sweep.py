import os
import sys
#sys.path.append('')
from dotenv import load_dotenv
import pandas as pd
from hybrid.sites.site_info  import SiteInfo
import time
import numpy as np

import warnings
warnings.filterwarnings("ignore")
# import yaml
from national_sweep_optimize import opt_national_sweep
# class call_prelim_opt:
# #Step 1: initialize inputs
# #     a: costs
# #     b: design
# #     c: load profile (opt)
#     def __init__(self):
def return_frac_of_sites(sitelistfilepath,start_idx,num_sites):
    df=pd.read_pickle(sitelistfilepath).iloc[start_idx:start_idx+num_sites]
    return df

hub_height = 100
num_sites = 1354
results_subdir ='100mHubHt_Try01'
parent_path = os.path.abspath('') + '/'
main_dir = parent_path + 'NATIONAL_SWEEP/'
default_input_file = main_dir + 'sweep_defaults.csv'
model_params = pd.read_csv(default_input_file,index_col = 'Variable')
model_params.loc['Hub Height']=hub_height
sitelistfilename = '100mSiteObjs_1354'

opt = opt_national_sweep(model_params)
n_run_at_a_time = 25
num_runs = num_sites//n_run_at_a_time
start_count = 425

for i in range(num_runs):
    start = time.perf_counter()
    site_df=return_frac_of_sites(main_dir + sitelistfilename,start_count,n_run_at_a_time)

    opt.run_all(site_df,hub_height,results_subdir)
    end = time.perf_counter()
    start_count +=n_run_at_a_time
    print("Completed run for chunk {}/{}Took {} minutes".format(i,num_runs,round(end - start, 3)))
    []
[]
        #50k points is 2km resolution
        #can use 10-50km resolution

        #west coast lon = -124.58
        #east coast lon = -67.924
        #northern lat: 49.54
        #southern lat: 25.62533
        
