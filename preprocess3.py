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

    search_ids = []

    # get all unique srch_id and iterate through them
    for search_id in df['srch_id'].unique():
        
        #Get the data for one search_id
        sdf = df[df['srch_id']==search_id]
        sdf = sdf.sort_values(by = ['position']) 

        #Computes the wieghtseach 
        weight = np.linspace(len(sdf),0,len(sdf))

        booked = sdf['booking_bool'].max() == 1
        clicked = sdf['click_bool'].max() == 1



        #Make an list with statistics for  search
        stat = []
        stat.append(search_id)    
        stat.append(len(sdf))
        if booked:
    	    stat.append(1)
        else:
            stat.append(0)
        if clicked:
            stat.append(1)
        else:
            stat.append(0)

        stat.append(sdf['visitor_location_country_id'].iloc[0])        # costumers country ID
        stat.append(sdf['visitor_hist_starrating'].iloc[0])            # history mean star rating (NaN else)
        stat.append(sdf['visitor_hist_adr_usd'].iloc[0])               # mean price earlier booked (NaN else)
        stat.append(sdf['prop_country_id'].iloc[0])                    # hotel country ID
        stat.append(sdf['srch_destination_id'].iloc[0])
        stat.append(sdf['srch_length_of_stay'].iloc[0])
        stat.append(sdf['srch_booking_window'].iloc[0])
        stat.append(sdf['srch_adults_count'].iloc[0])
        stat.append(sdf['srch_children_count'].iloc[0])
        stat.append(sdf['srch_room_count'].iloc[0])
        stat.append(sdf['srch_saturday_night_bool'].iloc[0])
        stat.append(sdf['random_bool'].iloc[0])
        
        stat_col1 = ['srch_id', 'rows', 'booked', 'clicked', 'visitor_location_country_id', 'visitor_hist_starrating',
            'visitor_hist_adr_usd', 'prop_country_id', 'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window',
            'srch_adults_count', 'srch_children_count', 'srch_room_count', 'srch_saturday_night_bool', 'random_bool']
        
        if booked:
          
            stat.append(sdf[sdf['booking_bool'] == 1]['prop_id'].iloc[0]) #Hotel ID
            stat.append(first(sdf[(sdf['booking_bool'] == 1) & (sdf['prop_starrating']!=0)]['prop_starrating']))
            stat.append(sdf[(sdf['booking_bool'] == 1) ]['prop_brand_bool'].iloc[0])
            stat.append(sdf[sdf['booking_bool'] == 1]['prop_location_score1'].iloc[0])
            stat.append(sdf[sdf['booking_bool'] == 1]['prop_location_score2'].iloc[0])
            stat.append(sdf[sdf['booking_bool'] == 1]['prop_review_score'].iloc[0])
            stat.append(sdf[sdf['booking_bool'] == 1]['prop_log_historical_price'].iloc[0])
            stat.append(sdf[sdf['booking_bool'] == 1]['position'].iloc[0])  
            stat.append(sdf[sdf['booking_bool'] == 1]['price_usd'].iloc[0])
            stat.append(sdf[sdf['booking_bool'] == 1]['promotion_flag'].iloc[0])
            stat.append(sdf[sdf['booking_bool'] == 1]['srch_query_affinity_score'].iloc[0])
            stat.append(sdf[sdf['booking_bool'] == 1]['orig_destination_distance'].iloc[0])
            stat.append(sdf[sdf['booking_bool'] == 1]['gross_bookings_usd'].iloc[0])

        elif clicked:
           
            stat.append(sdf[sdf['click_bool'] == 1]['prop_id'].iloc[0])
            stat.append(average(sdf[(sdf['click_bool'] == 1) & (sdf['prop_starrating']!=0) ]['prop_starrating'],weight,None))
            stat.append(np.round(np.average(sdf[sdf['click_bool'] == 1]['prop_brand_bool'])))
            stat.append(average(sdf[sdf['booking_bool'] == 1]['prop_location_score1'],weight,None))
            stat.append(average(sdf[sdf['booking_bool'] == 1]['prop_location_score2'],weight,None)) 
            stat.append(average(sdf[sdf['booking_bool'] == 1]['prop_review_score'],weight,0))
            stat.append(average(sdf[sdf['click_bool'] == 1]['prop_log_historical_price'],weight, 0))
            stat.append(sdf[sdf['click_bool'] == 1]['position'].iloc[0])
            stat.append(np.average(sdf[sdf['click_bool'] == 1]['price_usd'].iloc[0]))
            stat.append(np.round(np.average(sdf[sdf['click_bool'] == 1]['promotion_flag'])))
            stat.append(np.average(sdf[sdf['click_bool'] == 1]['srch_query_affinity_score']))
            stat.append(np.average(sdf[sdf['click_bool'] == 1]['orig_destination_distance']))
            stat.append(sdf[sdf['click_bool'] == 1]['gross_bookings_usd'].iloc[0])

        else:

            stat.append(sdf['prop_id'].iloc[0])
            stat.append(average(sdf['prop_starrating'], weight, 0))
            stat.append(np.round(average(sdf['prop_brand_bool'],weight,None)))
            stat.append(average(sdf['prop_location_score1'], weight, None))
            stat.append(average(sdf['prop_location_score2'], weight, None))
            stat.append(average(sdf['prop_review_score'], weight, 0))
            stat.append(average(sdf['prop_log_historical_price'], weight, 0))
            stat.append(np.nan)
            stat.append(np.average(sdf['price_usd']))
            stat.append(np.round(np.average(sdf['promotion_flag'])))
            stat.append(np.average(sdf['srch_query_affinity_score']))
            stat.append(np.average(sdf['orig_destination_distance']))
            stat.append(np.average(sdf['gross_bookings_usd']))  
            
        stat_col2 = ['prop_id', 'prop_starrating', 'prop_brand_bool', 'prop_location_score1', 'prop_location_score2',
            'prop_review_score', 'prop_log_historical_price', 'position', 'price_usd', 'promotion_flag', 'srch_query_affinity_score',
            'orig_destination_distance', 'gross_bookings_usd']
                
         # average columns
        stat.append(average(sdf['prop_log_historical_price'],weight, 0))
        stat.append(average(sdf['price_usd'],weight, None))
        stat.append(average(sdf['srch_query_affinity_score'],weight, None))
        stat.append(average(sdf['orig_destination_distance'],weight, None))
        stat.append(average(sdf['prop_starrating'],weight, 0))
        stat.append(average(sdf['prop_location_score1'],weight, None))
        stat.append(average(sdf['prop_location_score2'],weight, None))
        stat.append(average(sdf['prop_review_score'],weight, 0))           
                
        stat_col3 = ['prop_log_historical_price_avg', 'price_usd_avg', 'srch_query_affinity_score_avg',
            'orig_destination_distance_avg', 'prop_starrating_avg', 'prop_location_score1_avg', 
            'prop_location_score2_avg', 'srch_query_affinity_score_avg']            
                
        rate = ['comp1_rate', 'comp2_rate', 'comp3_rate', 'comp4_rate', 'comp5_rate',
        	 'comp6_rate', 'comp7_rate', 'comp8_rate']
        
        inv = ['comp1_inv', 'comp2_inv', 'comp3_inv', 'comp4_inv', 'comp5_inv', 'comp6_inv', 
                'comp7_inv', 'comp8_inv']
        
        diff = ['comp1_rate_percent_diff', 'comp2_rate_percent_diff', 'comp3_rate_percent_diff',
                'comp4_rate_percent_diff', 'comp5_rate_percent_diff', 'comp6_rate_percent_diff',
                'comp7_rate_percent_diff', 'comp8_rate_percent_diff']
    
        stat.append(sdf[rate].mean().mean()) # price competition (1=better, 0=none, -1=bad)
        stat.append(sdf[inv].mean().mean())  # availibility competition (1=better, 0=same)
        stat.append(sdf[diff].min().min())  # % differences price competition
        stat.append(sdf[diff].max().max())
    	
        stat_col4 = ['comp_rate', 'comp_inv', 'comp_rate_percent_diff_min', 'comp_rate_percent_diff_min']
        
        search_ids.append(pd.DataFrame(stat,index = stat_col1+stat_col2+stat_col3 + stat_col4))

    return pd.concat(search_ids,axis = 1).T


# function read in data and process chunks to combine afterwords
def make_data(filename, chunksize = 100000):
    new_data = []
    for df in pd.read_csv(filename, chunksize=chunksize):
        new_data.append(process(df))
    result = pd.concat(new_data, axis = 0)
    return result


if len(sys.argv) > 1: 
    filename = sys.argv[1]
else:
    print("specify filename plz")

new = make_data(filename, chunksize =100000)
new.to_csv('preprocessed.csv', index =False)

