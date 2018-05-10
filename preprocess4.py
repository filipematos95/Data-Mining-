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



# help functions to take average over a series
def average(feature, w, exclude = None):

    if exclude != None:
        exclude = feature[feature != exclude]
        if len(exclude) > 0:
            return np.average(exclude, weights=w[0:len(exclude)])
    else:
        if len(feature) > 0:
            return np.average(feature, weights=w[0:len(feature)])

    return np.nan


# function that returns nan if empty and elsethefirst element
def first(s):
    
    if s.empty:
        return np.nan
    else:
        return s.iloc[0]


# function processes chunks of read in data frame
def process(df):

    search_ids_p = []
    search_ids_n = []

    # get all unique srch_id and iterate through them
    for search_id in df['srch_id'].unique():
        
        #Get the data for one search_id
        sdf = df[df['srch_id'] == search_id]
        sdf = sdf.sort_values(by = ['position']) 

        # get example data of non booking
        sdf_n = sdf.copy().sort_values(by = ['position'], ascending = False) 
        
        if sdf_n[(sdf_n['booking_bool']==0) & (sdf_n['click_bool']==0)].empty:
            if sdf_n[(sdf_n['click_bool']==0)].empty == False:
                sdf_n = sdf_n[(sdf_n['click_bool']==0)]
        else:
            sdf_n = sdf_n[(sdf_n['booking_bool']==0) & (sdf_n['click_bool']==0)]        
        
        #Computes the wieghtseach 
        weight = np.linspace(len(sdf),0,len(sdf))
        weight_n = np.linspace(len(sdf_n),0,len(sdf_n))

        booked = sdf['booking_bool'].max() == 1
        clicked = sdf['click_bool'].max() == 1

        #Make an list with statistics for  search
        stat_p = []
        stat_p.append(search_id)    
        stat_p.append(len(sdf))
        
        stat_n = []
        stat_n.append(-search_id)    
        stat_n.append(len(sdf_n))
        
        # booked
        if booked:
    	    stat_p.append(1)
    	    stat_n.append(0)
        else:
            stat_p.append(0)
            stat_n.append(0)
        # clicked
        if clicked:
            stat_p.append(1)
            stat_n.append(0)
        else:
            stat_p.append(0)
            stat_n.append(0)

        # positive
        stat_p.append(sdf['visitor_location_country_id'].iloc[0])        # costumers country ID
        stat_p.append(sdf['visitor_hist_starrating'].iloc[0])            # history mean star rating (NaN else)
        stat_p.append(sdf['visitor_hist_adr_usd'].iloc[0])               # mean price earlier booked (NaN else)
        stat_p.append(sdf['prop_country_id'].iloc[0])                    # hotel country ID
        stat_p.append(sdf['srch_destination_id'].iloc[0])
        stat_p.append(sdf['srch_length_of_stay'].iloc[0])
        stat_p.append(sdf['srch_booking_window'].iloc[0])
        stat_p.append(sdf['srch_adults_count'].iloc[0])
        stat_p.append(sdf['srch_children_count'].iloc[0])
        stat_p.append(sdf['srch_room_count'].iloc[0])
        stat_p.append(sdf['srch_saturday_night_bool'].iloc[0])
        stat_p.append(sdf['random_bool'].iloc[0])
        
        # negative (first is last, order is reveresed)
        stat_n.append(sdf_n['visitor_location_country_id'].iloc[0])        # costumers country ID
        stat_n.append(sdf_n['visitor_hist_starrating'].iloc[0])            # history mean star rating (NaN else)
        stat_n.append(sdf_n['visitor_hist_adr_usd'].iloc[0])               # mean price earlier booked (NaN else)
        stat_n.append(sdf_n['prop_country_id'].iloc[0])                    # hotel country ID
        stat_n.append(sdf_n['srch_destination_id'].iloc[0])
        stat_n.append(sdf_n['srch_length_of_stay'].iloc[0])
        stat_n.append(sdf_n['srch_booking_window'].iloc[0])
        stat_n.append(sdf_n['srch_adults_count'].iloc[0])
        stat_n.append(sdf_n['srch_children_count'].iloc[0])
        stat_n.append(sdf_n['srch_room_count'].iloc[0])
        stat_n.append(sdf_n['srch_saturday_night_bool'].iloc[0])
        stat_n.append(sdf_n['random_bool'].iloc[0])
        
        stat_col1 = ['srch_id', 'rows', 'booked', 'clicked', 'visitor_location_country_id', 'visitor_hist_starrating',
            'visitor_hist_adr_usd', 'prop_country_id', 'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window',
            'srch_adults_count', 'srch_children_count', 'srch_room_count', 'srch_saturday_night_bool', 'random_bool','booked_&_clicked']
        
        if booked:
            
            # positive
            stat_p.append(2)
            stat_p.append(sdf[sdf['booking_bool'] == 1]['prop_id'].iloc[0])
            stat_p.append(first(sdf[(sdf['booking_bool'] == 1) & (sdf['prop_starrating']!=0)]['prop_starrating']))
            stat_p.append(sdf[(sdf['booking_bool'] == 1) ]['prop_brand_bool'].iloc[0])
            stat_p.append(sdf[sdf['booking_bool'] == 1]['prop_location_score1'].iloc[0])
            stat_p.append(sdf[sdf['booking_bool'] == 1]['prop_location_score2'].iloc[0])
            stat_p.append(sdf[sdf['booking_bool'] == 1]['prop_review_score'].iloc[0])
            stat_p.append(sdf[sdf['booking_bool'] == 1]['prop_log_historical_price'].iloc[0])
            stat_p.append(sdf[sdf['booking_bool'] == 1]['position'].iloc[0])  
            stat_p.append(sdf[sdf['booking_bool'] == 1]['price_usd'].iloc[0])
            stat_p.append(sdf[sdf['booking_bool'] == 1]['promotion_flag'].iloc[0])
            stat_p.append(sdf[sdf['booking_bool'] == 1]['srch_query_affinity_score'].iloc[0])
            stat_p.append(sdf[sdf['booking_bool'] == 1]['orig_destination_distance'].iloc[0])
            stat_p.append(sdf[sdf['booking_bool'] == 1]['gross_bookings_usd'].iloc[0])

        elif clicked:
            
            # positive
            stat_p.append(1)
            stat_p.append(sdf[sdf['click_bool'] == 1]['prop_id'].iloc[0])
            stat_p.append(average(sdf[(sdf['click_bool'] == 1) & (sdf['prop_starrating']!=0) ]['prop_starrating'],weight,None))
            stat_p.append(np.round(np.average(sdf[sdf['click_bool'] == 1]['prop_brand_bool'])))
            stat_p.append(average(sdf[sdf['booking_bool'] == 1]['prop_location_score1'],weight,None))
            stat_p.append(average(sdf[sdf['booking_bool'] == 1]['prop_location_score2'],weight,None)) 
            stat_p.append(average(sdf[sdf['booking_bool'] == 1]['prop_review_score'],weight,0))
            stat_p.append(average(sdf[sdf['click_bool'] == 1]['prop_log_historical_price'],weight, 0))
            stat_p.append(sdf[sdf['click_bool'] == 1]['position'].iloc[0])
            stat_p.append(np.average(sdf[sdf['click_bool'] == 1]['price_usd'].iloc[0]))
            stat_p.append(np.round(np.average(sdf[sdf['click_bool'] == 1]['promotion_flag'])))
            stat_p.append(np.average(sdf[sdf['click_bool'] == 1]['srch_query_affinity_score']))
            stat_p.append(np.average(sdf[sdf['click_bool'] == 1]['orig_destination_distance']))
            stat_p.append(sdf[sdf['click_bool'] == 1]['gross_bookings_usd'].iloc[0])
            
        else:
            
            # positive (?)
            stat_p.append(0)
            stat_p.append(sdf['prop_id'].iloc[0])
            stat_p.append(average(sdf['prop_starrating'], weight, 0))
            stat_p.append(np.round(average(sdf['prop_brand_bool'],weight,None)))
            stat_p.append(average(sdf['prop_location_score1'], weight, None))
            stat_p.append(average(sdf['prop_location_score2'], weight, None))
            stat_p.append(average(sdf['prop_review_score'], weight, 0))
            stat_p.append(average(sdf['prop_log_historical_price'], weight, 0))
            stat_p.append(np.nan)
            stat_p.append(np.average(sdf['price_usd']))
            stat_p.append(np.round(np.average(sdf['promotion_flag'])))
            stat_p.append(np.average(sdf['srch_query_affinity_score']))
            stat_p.append(np.average(sdf['orig_destination_distance']))
            stat_p.append(np.average(sdf['gross_bookings_usd']))  
          
         
        # negative
        stat_n.append(0)
        stat_n.append(sdf_n['prop_id'].iloc[0])
        stat_n.append(first(sdf_n[(sdf_n['prop_starrating']!=0)]['prop_starrating']))
        stat_n.append(sdf_n['prop_brand_bool'].iloc[0])
        stat_n.append(sdf_n['prop_location_score1'].iloc[0])
        stat_n.append(sdf_n['prop_location_score2'].iloc[0])
        stat_n.append(sdf_n['prop_review_score'].iloc[0])
        stat_n.append(sdf_n['prop_log_historical_price'].iloc[0])
        stat_n.append(sdf_n['position'].iloc[0])  
        stat_n.append(sdf_n['price_usd'].iloc[0])
        stat_n.append(sdf_n['promotion_flag'].iloc[0])
        stat_n.append(sdf_n['srch_query_affinity_score'].iloc[0])
        stat_n.append(sdf_n['orig_destination_distance'].iloc[0])
        stat_n.append(sdf_n['gross_bookings_usd'].iloc[0]) 

        stat_col2 = ['prop_id', 'prop_starrating', 'prop_brand_bool', 'prop_location_score1', 'prop_location_score2',
            'prop_review_score', 'prop_log_historical_price', 'position', 'price_usd', 'promotion_flag', 'srch_query_affinity_score',
            'orig_destination_distance', 'gross_bookings_usd']
                
         # average columns
        # stat_p.append(average(sdf['prop_log_historical_price'],weight, 0))
        # stat_p.append(average(sdf['price_usd'],weight, None))
        # stat_p.append(average(sdf['srch_query_affinity_score'],weight, None))
        # stat_p.append(average(sdf['orig_destination_distance'],weight, None))
        # stat_p.append(average(sdf['prop_starrating'],weight, 0))
        # stat_p.append(average(sdf['prop_location_score1'],weight, None))
        # stat_p.append(average(sdf['prop_location_score2'],weight, None))
        # stat_p.append(average(sdf['prop_review_score'],weight, 0))           
                
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
        stat_p.append(sdf[rate].mean().mean()) # price competition (1=better, 0=none, -1=bad)
        stat_p.append(sdf[inv].mean().mean())  # availibility competition (1=better, 0=same)
        stat_p.append(sdf[diff].min().min())  # % differences price competition
        stat_p.append(sdf[diff].max().max())

        # negatives
        stat_n.append(sdf_n[rate].mean().mean()) # price competition (1=better, 0=none, -1=bad)
        stat_n.append(sdf_n[inv].mean().mean())  # availibility competition (1=better, 0=same)
        stat_n.append(sdf_n[diff].min().min())  # % differences price competition
        stat_n.append(sdf_n[diff].max().max())
    	
        stat_col4 = ['comp_rate', 'comp_inv', 'comp_rate_percent_diff_min', 'comp_rate_percent_diff_max']
        
        search_ids_p.append(pd.DataFrame(stat_p,index = stat_col1+stat_col2+stat_col4)) # + stat_col3))
        search_ids_n.append(pd.DataFrame(stat_n,index = stat_col1+stat_col2+stat_col4)) # + stat_col3))
        total_search_id = search_ids_p + search_ids_n

    return pd.concat(total_search_id,axis = 1).T


# function read in data and process chunks to combine afterwords
def make_data(filename, chunksize = 100000):
    
    search_ids = []

    for df in pd.read_csv(filename, chunksize=chunksize):
        search_ids.append(process(df))

    # orange row
    meta1 = ['d','c','d','d','d','c','c','d','d','c','c','c','c','c','d','d','d'] # discrete (d), continuous (c), string (s)
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
new.to_csv(filename[:-4]+'_preprocessed4.csv', index =False)


