#########################################
#                                       #
#               Group 24                #
#               (2018)                  #
#           Vrije Universiteit          #
#          preprocess testdata          #
#                                       #
#########################################

import itertools
import pandas as pd
import numpy as np
import sys
import copy as copy
from multiprocessing import Pool


"""
File reads in data by chunks to compress search id to one row.
"""


def booked_clicked(sdf,means):
    stat_p = []
    stat_p.append(sdf['srch_id'])   
    stat_p.append(sdf['booking_bool']) 
    stat_p.append(sdf['click_bool']) 
    stat_p.append(sdf['visitor_location_country_id'])        
    stat_p.append(sdf['visitor_hist_starrating'])            
    stat_p.append(sdf['visitor_hist_adr_usd'])               
    stat_p.append(sdf['prop_country_id'])                    
    stat_p.append(sdf['srch_destination_id'])
    stat_p.append(sdf['srch_length_of_stay'])
    stat_p.append(sdf['srch_booking_window'])
    stat_p.append(sdf['srch_adults_count'])
    stat_p.append(sdf['srch_children_count'])
    stat_p.append(sdf['srch_room_count'])
    stat_p.append(sdf['srch_saturday_night_bool'])
    stat_p.append(sdf['random_bool'])
    stat_p.append(sdf['prop_id'])

    if np.isnan(sdf['prop_starrating']) or sdf['prop_starrating'] == 0: 
        stat_p.append(means[0])
    else:
        stat_p.append(sdf['prop_starrating'])

    stat_p.append(sdf['prop_brand_bool'])

    if np.isnan(sdf['prop_location_score1']):
        stat_p.append(means[3])
    else:
        stat_p.append(sdf['prop_location_score1'])

    if np.isnan(sdf['prop_location_score2']):
        stat_p.append(means[2])
    else:
        stat_p.append(sdf['prop_location_score2'])

    if np.isnan(sdf['prop_review_score']) or sdf['prop_review_score'] == 0:
        stat_p.append(means[1])
    else:
        stat_p.append(sdf['prop_review_score'])
    
    stat_p.append(sdf['prop_log_historical_price'])
    stat_p.append(sdf['position'])  
    stat_p.append(sdf['price_usd'])
    stat_p.append(sdf['promotion_flag'])
    stat_p.append(sdf['srch_query_affinity_score'])
    stat_p.append(sdf['orig_destination_distance'])
    stat_p.append(sdf['gross_bookings_usd'])

    stat_p.append(sdf['rate_sum']) # price competition (1=better, 0=none, -1=bad)
    stat_p.append(sdf['inv_sum'])
    stat_p.append(sdf['diff_mean'])
    stat_p.append(sdf['rate_abs']) # 0 if there is no comp with lower price otherwise 1
    stat_p.append(sdf['inv_abs']) # 
    stat_p.append(sdf['ump']) 
    stat_p.append(sdf['price_diff'])
    stat_p.append(sdf['starrating_diff'])
    stat_p.append(sdf['per_fee'])
    stat_p.append(sdf['score2ma'])
    stat_p.append(sdf['total_fee'])
    stat_p.append(sdf['score1d2'])
    stat_p.append(sdf['hotel_quality_1'])
    stat_p.append(sdf['hotel_quality_2'])

    return stat_p

# function processes chunks of read in data frame
def process(df):
    
    search_ids_p = []

    stat_col1 = ['srch_id', 'booked', 'clicked', 'visitor_location_country_id', 'visitor_hist_starrating',
            'visitor_hist_adr_usd', 'prop_country_id', 'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window',
            'srch_adults_count', 'srch_children_count', 'srch_room_count', 'srch_saturday_night_bool', 'random_bool']
          
         
    stat_col2 = ['prop_id', 'prop_starrating', 'prop_brand_bool', 'prop_location_score1', 'prop_location_score2',
            'prop_review_score', 'prop_log_historical_price', 'position', 'price_usd', 'promotion_flag', 'srch_query_affinity_score',
            'orig_destination_distance', 'gross_bookings_usd']

    stat_col4 = ['rate_sum', 'inv_sum', 'diff_mean', 'rate_abs','inv_abs']    

    stat_col5 = ['ump','price_diff','starrating_diff','per_fee','score2ma','total_fee','score1d2','hotel_quality_1','hotel_quality_2']

    # get all unique src
    search_ids = []
    for search_id in df['srch_id'].unique():

        # Get the data for one search_id
        sdf = df[df['srch_id'] == search_id]
        
        # Computes the weights 
        prop_starrating_mean = sdf['prop_starrating'].mean()
        prop_review_score_mean = sdf['prop_review_score'].mean()
        prop_location_score2_mean = sdf['prop_location_score2'].mean()
        prop_location_score1_mean = sdf['prop_location_score1'].mean()

        means = [prop_starrating_mean,prop_review_score_mean,prop_location_score2_mean,prop_location_score1_mean]

        # Make an list with statistics for  search
        for index, sdf in sdf.iterrows():
            stat = booked_clicked(sdf,means)
            search_ids.append(pd.DataFrame(stat,index = stat_col1+stat_col2+stat_col4+stat_col5)) 

        search_ids.append(pd.DataFrame(stat,index = stat_col1+stat_col2+stat_col4+stat_col5))    
        #for index,sdf in neg.iterrows():

    return pd.concat(search_ids,axis = 1).T

# function read in data and process chunks to combine afterwords
def make_data(filename, chunksize = 100000):
    pool = Pool(3)

    df_split =  pd.read_csv(filename, chunksize = chunksize,engine = 'c')
    result = pd.concat(pool.map(process,df_split))

    pool.close()
    pool.join()
        
    return result

chunksize = 100000
if len(sys.argv) > 1: 
    filename = sys.argv[1]
    if len(sys.argv) > 2: 
        chunksize = int(sys.argv[2])

else:
    print("Please specify the filename")

new = make_data(filename, chunksize = chunksize)
new.to_csv(filename[:-4]+'_preprocessed.csv', index =False)


