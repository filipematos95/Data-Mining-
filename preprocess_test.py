#########################################
#                                       #
#               Group 24                #
#               (2018)                  #
#           Vrije Universiteit          #
#           data exploration            #
#                                       #
#########################################

import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import copy as copy
#%matplotlib inline



"""
File reads in data by chunks to compress search id to one row.
"""

###################################### readin data ########################################


# function processes chunks of read in data frame
def process(df):

    
    search_ids_p = []

    # get all unique srch_id and iterate through them
    for index, sdf in df.iterrows():        
        processed = []
        # positive
        processed.append(sdf['srch_id'])

        if sdf['booking_bool'] == 1:
            processed.append(1)
            booked = 1
        else:
            processed.append(0)
            booked = 0
        # clicked
        if sdf['click_bool'] == 1:
            processed.append(1)
            clicked = 1
        else:
            processed.append(0)
            clicked = 0

        processed.append(sdf['visitor_location_country_id'])        # costumers country ID
        processed.append(sdf['visitor_hist_starrating'])            # history mean star rating (NaN else)
        processed.append(sdf['visitor_hist_adr_usd'])               # mean price earlier booked (NaN else)
        processed.append(sdf['prop_country_id'])                    # hotel country ID
        processed.append(sdf['srch_destination_id'])
        processed.append(sdf['srch_length_of_stay'])
        processed.append(sdf['srch_booking_window'])
        processed.append(sdf['srch_adults_count'])
        processed.append(sdf['srch_children_count'])
        processed.append(sdf['srch_room_count'])
        processed.append(sdf['srch_saturday_night_bool'])
        processed.append(sdf['random_bool'])
        
        stat_col1 = ['srch_id','booked','clicked','visitor_location_country_id','visitor_hist_starrating',
            'visitor_hist_adr_usd', 'prop_country_id', 'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window',
            'srch_adults_count', 'srch_children_count', 'srch_room_count', 'srch_saturday_night_bool','random_bool','booked_&_clicked']
        
        if booked or clicked:
            processed.append(1)
        else: 
            processed.append(0)

        processed.append(sdf['prop_id'])
        processed.append(sdf['prop_starrating'])
        processed.append(sdf['prop_brand_bool'])
        processed.append(sdf['prop_location_score1'])
        processed.append(sdf['prop_location_score2'])
        processed.append(sdf['prop_review_score'])
        processed.append(sdf['prop_log_historical_price'])
        processed.append(sdf['position'])  
        processed.append(sdf['price_usd'])
        processed.append(sdf['promotion_flag'])
        processed.append(sdf['srch_query_affinity_score'])
        processed.append(sdf['orig_destination_distance'])
        processed.append(sdf['gross_bookings_usd']) 
         
        stat_col2 = ['prop_id', 'prop_starrating', 'prop_brand_bool', 'prop_location_score1', 'prop_location_score2',
            'prop_review_score', 'prop_log_historical_price', 'position', 'price_usd', 'promotion_flag', 'srch_query_affinity_score',
            'orig_destination_distance', 'gross_bookings_usd']
                
         # average columns
        # processed.append(average(sdf['prop_log_historical_price'],weight, 0))
        # processed.append(average(sdf['price_usd'],weight, None))
        # processed.append(average(sdf['srch_query_affinity_score'],weight, None))
        # processed.append(average(sdf['orig_destination_distance'],weight, None))
        # processed.append(average(sdf['prop_starrating'],weight, 0))
        # processed.append(average(sdf['prop_location_score1'],weight, None))
        # processed.append(average(sdf['prop_location_score2'],weight, None))
        # processed.append(average(sdf['prop_review_score'],weight, 0))           
                
        # stat_col3 = ['prop_log_historical_price_avg', 'price_usd_avg', 'srch_query_affinity_score_avg',
        #     'orig_destination_distance_avg', 'prop_starrating_avg', 'prop_location_score1_avg', 
        #     'prop_location_score2_avg', 'srch_query_affinity_score_avg']            
                
        rate = ['comp1_rate', 'comp2_rate', 'comp3_rate', 'comp4_rate', 'comp5_rate',
        	 'comp6_rate', 'comp7_rate', 'comp8_rate']
        
        inv = ['comp1_inv', 'comp2_inv', 'comp3_inv', 'comp4_inv', 'comp5_inv', 'comp6_inv', 
                'comp7_inv', 'comp8_inv']
        
        diff = ['comp1_rate_percent_diff', 'comp2_rate_percent_diff', 'comp3_rate_percent_diff',
                'comp4_rate_percent_diff', 'comp5_rate_percent_diff', 'comp6_rate_percent_diff',
                'comp7_rate_percent_diff', 'comp8_rate_percent_diff']
    
        # positives
        processed.append(sdf[rate].mean()) # price competition (1=better, 0=none, -1=bad)
        processed.append(sdf[inv].mean())  # availibility competition (1=better, 0=same)
        processed.append(sdf[diff].min())  # % differences price competition
        processed.append(sdf[diff].max())
    	
        stat_col4 = ['comp_rate', 'comp_inv', 'comp_rate_percent_diff_min', 'comp_rate_percent_diff_max']
        
        search_ids_p.append(pd.DataFrame(processed,index = stat_col1+stat_col2+stat_col4)) # + stat_col3))
    total_search_id = search_ids_p 

    return pd.concat(total_search_id,axis = 1).T


# function read in data and process chunks to combine afterwords
def make_data(filename, chunksize = 100000):
    
    search_ids = []

    for df in pd.read_csv(filename, chunksize=chunksize):
        search_ids.append(process(df))

    # orange row
    meta1 = ['d','c','c','d','c','c','d','d','c','c','c','c','c','d','c','d'] # discrete (d), continuous (c), string (s)
    meta2 = ['d','c','d','c','c','c','c','c','c', 'd','c', 'c','c']    
    #meta3 = ['c','c','c','c','c','c','c','c']
    meta4 = ['c','c','c','c']
    
    index = pd.DataFrame(meta1 + meta2 + meta4, index= search_ids[0].columns).T
    extra = index.copy()
    extra[extra != np.nan] = np.nan
    extra.iloc[0,2] = 'c'
    total = [index] + [extra] + search_ids
        
    result = pd.concat(total, axis = 0)
    return result


chunksize = 100000
if len(sys.argv) > 1: 
    filename = sys.argv[1]
    if len(sys.argv) > 2: 
        chunksize = int(sys.argv[2])

else:
    print("specify filename plz")


new = make_data(filename, chunksize = chunksize)
new.to_csv(filename[:-4]+'_preprocessed.csv', index =False)


