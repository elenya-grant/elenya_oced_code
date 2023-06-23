import os
import sys
#sys.path.append('')
from dotenv import load_dotenv
import pandas as pd
from hybrid.sites.site_info  import SiteInfo

import numpy as np

import warnings
# from pathlib import Path
# import time
from analysis import run_hybrid_plant
from scipy import interpolate
from tools.floris_2_csv import csv_from_floris
#from math import cos, asin, sqrt, pi
warnings.filterwarnings("ignore")
# import yaml


        #west coast lon = -124.58
        #east coast lon = -67.924
        #northern lat: 49.54
        #southern lat: 25.62533
        

parent_path = os.path.abspath('') + '/'
main_dir = parent_path + 'NATIONAL_SWEEP/'

df=pd.read_csv(main_dir + '100mSiteInfo.csv')
id_val,id_cnt=np.unique(df['Site ID'].values,return_counts = True)
id_idx_duplicates=np.argwhere(id_cnt>1)[:,0]
id_for_mult=id_val[id_idx_duplicates]
idx_to_drop=[]
for id in id_for_mult:
    idx=df.index[df['Site ID']==id]
    idx_to_drop.append(idx[1:])

    #1354 sites without duplicates
# dfobj=pd.read_pickle(main_dir + '100mSiteObjs')
for i in range(len(idx_to_drop)):
    df=df.drop(idx_to_drop[i],axis=0)
    # dfobj=dfobj.drop(idx_to_drop[i],axis=0)
[]

do_other_stuff=False
if do_other_stuff:
    wtk_file = main_dir + 'wtk_site_metadata.csv'
    resource_directory = parent_path + 'resource_files/'
    resource_year = 2013
    # locs = pd.read_csv(wtk_file)
    locs = pd.read_csv(main_dir + 'wtk_simple_sites_7kadj_onshore.csv')
    # locs = pd.read_csv(main_dir + 'wtk_simple_sites_7kadj_onshore_enoughArea.csv')
    # sample_site = pd.read_pickle(parent_path + 'hybrid/sites/sample_site').to_dict()

    []

    def make_site(latitude,longitude,sample_site):
        #print("[3]: making site...")
        
        sample_site['lat']=latitude
        sample_site['lon']=longitude
        sample_site['year']= resource_year
        sample_site['no_wind'] = False
        sample_site['no_solar'] = False
        #is below step necessary?
        
        site = SiteInfo(sample_site,resource_dir = resource_directory,hub_height=100)
        #self.main_dict['site']=site
        return site

    def get_100m_locs():
        sites=[]
        wind_files = os.listdir(resource_directory + 'wind/')
        # solar_files = os.listdir(resource_directory + 'solar/')
        w_lats = [float(w.split('_')[0]) for w in wind_files if '2013_60min_100m.srw' in w]
        w_lons = [float(w.split('_')[1]) for w in wind_files if '2013_60min_100m.srw' in w]
        sample_site = pd.read_pickle(parent_path + 'hybrid/sites/sample_site').to_dict()
        # s_lats = [float(s.split('_')[0]) for s in solar_files]
        # s_lons = [float(s.split('_')[1]) for s in solar_files]
        lons_list=locs['longitude'].values
        lats_list = locs['latitude'].values
        idx_list=[]
        for li,lat in enumerate(w_lats):
            # print(li)
            # # lat_error=(lats_list - np.round(lat,6))
            # # lon_error=(lons_list - np.round(w_lons[li],6))
            # site=make_site(lat,w_lons[li],sample_site)
            # sites.append(site)
            lat_error=(lats_list - lat)
            lon_error=(lons_list - w_lons[li])
            idx_lons = np.argwhere(np.abs(lon_error)<1)[:,0]
            idx_lats = np.argwhere(np.abs(lat_error)<1)[:,0]
            idx=[l for l in idx_lons if l in idx_lats]
            lat_smallest_error_idx=np.abs(lat_error[idx]).argmin()
            lon_smallest_error_idx=np.abs(lon_error[idx]).argmin()
            if np.abs(lat_error[idx[lat_smallest_error_idx]])<np.abs(lon_error[idx[lon_smallest_error_idx]]):
                final_idx = idx[lat_smallest_error_idx]
            else:
                final_idx = idx[lon_smallest_error_idx]
            idx_list.append(final_idx)

        return idx_list,w_lats,w_lons #,sites
        # sites=[]
        # for i,lat in enumerate(w_lats):
        #     site=make_site(lat,w_lons[i],sample_site)
        #     sites.append(site)
        
        []


        []


    idx_list,my_lats,my_lons,site_obj = get_100m_locs()
    df=pd.DataFrame({'Lat':my_lats,'Lon':my_lons,'State':locs['State'].iloc[idx_list].values,'cap_fac':locs['capacity_factor'].iloc[idx_list].values,'Site ID':locs['site_id'].iloc[idx_list].values})
    # df.to_pickle('100mSiteInfo')
    df.to_csv(main_dir + '100mSiteInfo.csv')
    df_obj=pd.DataFrame({'Lat':my_lats,'Lon':my_lons,'State':locs['State'].iloc[idx_list].values,'cap_fac':locs['capacity_factor'].iloc[idx_list].values,'Site Obj':site_obj,'Site ID':locs['site_id'].iloc[idx_list].values})
    df_obj.to_pickle(main_dir + '100mSiteObjs')
    def get_resource_info(state_info,state,day_num):
        site_cnt=[]
        #indexes=np.arange(start_idx,start_idx+num_calls+1,1)
        sites=[]
        lats=[]
        lons=[]
        ids=[]
        indexes = list(state_info.index.values)
        for i in indexes:
            # lat=self.locs.iloc[i]['latitude']
            # lon=self.locs.iloc[i]['longitude']
            # id=self.locs.iloc[i]['site_id']
            lat=state_info.loc[i]['latitude']
            lon=state_info.loc[i]['longitude']
            id=state_info.loc[i]['site_id']
            site=make_site(lat,lon,sample_site)
            lats.append(lat)
            lons.append(lon)
            ids.append(id)
            sites.append(site)

        print("Last index was {} for site ID {}".format(i,id))
        df=pd.DataFrame({'Site ID':ids,'Lat':lats,'Lon':lons,'Site Obj':site})
        # df.to_pickle(self.main_dir + 'Sites7k_{}_day{}'.format(state,day_num))

        return df

        

        





