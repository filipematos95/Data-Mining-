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
#%matplotlib inline


"""
File reads in data by chunks to compress search id to one row.

"""


###################################### readin data ########################################


# function processes chinks of read in data frame
def process(df):

    search_ids = []

    # get all unique srch_id and iterate through them
    for search_id in df['srch_id'].unique():

        #todo
        weight = np.linspace(len(sdf),0,len(sdf))
        
        # get the data for one search
        sdf = df[df['srch_id']==search_id]
        sdf = sdf.sort(['position'], ascending = 1) 
        
        # make an list with statistics for each search
        stat = []
        stat.append( search_id )    
        stat.append( len(sdf) )                                         # number of rows
        stat.append( sdf['visitor_location_country_id'].iloc[0])        # costumers country ID
        stat.append( sdf['visitor_hist_starrating'].iloc[0])            # history mean star rating (NaN else)
        stat.append( sdf['visitor_hist_adr_usd'].iloc[0])               # mean price earlier booked (NaN else)
        stat.append( sdf['prop_country_id'].iloc[0])                    # hotel country ID
        
        if sdf['booking_bool'].max() > 0:
            stat.append( sdf[sdf['booking_bool']>1]['prop_id'])         # hotel ID
        else:
            if sdf['click_bool'].max() > 0:)
                stat.append( sdf[sdf['click_bool']>1]['prop_id'].iloc[0])         # hotel ID
            else:
                stat.append( sdf['prop_id'].iloc[0])
        
        if sdf['booking_bool'].max() > 0:
            stat.append( sdf[ (sdf['booking_bool']>1) & (sdf['prop_starrating']!=0)]['prop_starrating']))
        elif  sdf['click_bool'].max() > 0:)
            stat.append( sdf[(sdf['click_bool']>1) & (sdf['prop_starrating']!=0) ]['prop_starrating'].iloc[0])         # prop_starrating 
        else:
            if len(avg) > 0:
                w = weigth[0:len(sdf[sdf['prop_starrating']>0]
                stat.append( np.average(sdf[(sdf['prop_starrating']!=0)],weights=w)
            else:
                stat.append(np.nan)
        avg = sdf[sdf['prop_starrating']>0]
        if len(avg) > 0:
            w = weigth[0:len(sdf[sdf['prop_starrating']>0]
            stat.append( np.average(sdf[sdf['prop_starrating']>0], weights=w])) # hotel star rating (costumer visists)(0 = missing) prop_starrating_avg
        else:
            stat.append(np.nan)
            
        avg = sdf[sdf['prop_review_score']>0]
        if len(avg) > 0:
            w = weigth[0:len(sdf[sdf['prop_review_score']>0]
            stat.append( np.average(sdf[sdf['prop_review_score']>0], weights=w])) # hotel star rating (0 = no revieuw, null = no info))(0 = missing) prop_starrating_avg
        else:
            stat.append(np.nan)
        
        stat.append( sdf[sdf['prop_review_score']>0]['prop_review_score'].mean())  # hotel star rating (0 = no revieuw, null = no info)
        
        if sdf['booking_bool'].max() > 0:
            stat.append( sdf[ (sdf['booking_bool']>1) ]['prop_brand_bool']))
        elif  sdf['click_bool'].max() > 0:)
            stat.append( sdf[(sdf['click_bool']>1) ]['prop_brand_bool'].iloc[0])         # prop_brand_bool large company? 
        else:
            stat.append( np.average(sdf['prop_brand_bool'], weights=weights)
        
        stat.append( np.average(sdf[(sdf) ], weights=weights)                          #  prop_brand_bool_avg large company? 
        
        
        stat_col1 = ['srch_id', 'rows', 'visitor_location_country_id', 'visitor_hist_starrating',
            'visitor_hist_adr_usd', 'prop_country_id', 'prop_id', 'prop_starrating', 'prop_starrating_avg'
            'prop_review_score', 'prop_review_score_avg', 'prop_brand_bool', 'prop_brand_bool_avg' ] 
        
        # needs to be fixed
        if sdf['booking_bool'].max() > 0:
            stat.append( sdf[sdf['booking_bool']>1]['prop_location_score1'])  # hotel location desirability (score1)
        else:
            stat.append( sdf['prop_location_score1'].mean())             # hotel location desirability (score1)

        if sdf['booking_bool'].max() > 0:
            stat.append( sdf[sdf['booking_bool']>1]['prop_location_score2'])  # hotel location desirability (score2)
        else:
            stat.append( sdf['prop_location_score2'].mean())            # hotel location desirability (score2)            
        
        
        stat.append( sdf[sdf['prop_log_historical_price']>0]['prop_log_historical_price'].mean())            # log of mean price when sold (not sold= 0)
        stat.append( sdf['position'].median())                           # hotel country ID (ONLY IN TRAINSET)
        stat.append( sdf['price_usd'].median())                          # hotel price (differ by country)
        stat.append( sdf['promotion_flag'].mean())                       # hotel sale price promotion(bool)
        stat.append( len(sdf['srch_destination_id'].unique())/len(sdf))  # destination hotel search ID
        stat.append( sdf['srch_length_of_stay'].median())                # search night stay
        stat.append( sdf['srch_booking_window'].median())                # seacrh booking window
        stat.append( sdf['srch_adults_count'].median())                  # number of adults
        stat.append( sdf['srch_children_count'].median())                # number of children
        stat.append( sdf['srch_room_count'].median())                    # hotel rooms
        stat.append( sdf['srch_saturday_night_bool'].mean())             # saterday included?
        stat.append( sdf['srch_query_affinity_score'].mean())            # The log(p) a hotel clicked internet (0 no register)
        stat.append( sdf['orig_destination_distance'].mean())            # Physical distance hotel-customer (0 = no calculations)
        stat.append( sdf['random_bool'].iloc[0])                         # random search order displayed? (INTERESTING)
        
        stat_col2 = ['prop_location_score1', 'prop_location_score2', 'prop_log_historical_price',
            'position', 'price_usd', 'promotion_flag', 'srch_destination_id', 'srch_length_of_stay',
            'srch_booking_window', 'srch_adults_count', 'srch_children_count', 'srch_room_count',
            'srch_saturday_night_bool', 'srch_query_affinity_score', 'orig_destination_distance',
            'random_bool']         

        stat.append( sdf['comp1_rate'].mean())                           # price competition (1=better, 0=none, -1=bad)
        stat.append( sdf['comp1_inv'].mean())                            # availibility competition (1=better, 0=same)
        stat.append( sdf['comp1_rate_percent_diff'].mean())              # % differences price competition
        stat.append( sdf['comp2_rate'].mean())                           # same:
        stat.append( sdf['comp2_inv'].mean())                            # 
        stat.append( sdf['comp2_rate_percent_diff'].mean())              # 
        stat.append( sdf['comp3_rate'].mean())                           #
        stat.append( sdf['comp3_inv'].mean())                            # 
        stat.append( sdf['comp3_rate_percent_diff'].mean())              # 
        stat.append( sdf['comp4_rate'].mean())                           #
        stat.append( sdf['comp4_inv'].mean())                            # 
        stat.append( sdf['comp4_rate_percent_diff'].mean())              # 
        stat.append( sdf['comp5_rate'].mean())                           # 
        stat.append( sdf['comp5_inv'].mean())                            # 
        stat.append( sdf['comp5_rate_percent_diff'].mean())              # 
        stat.append( sdf['comp6_rate'].mean())                           # 
        stat.append( sdf['comp6_inv'].mean())                            # 
        stat.append( sdf['comp6_rate_percent_diff'].mean())              # 
        stat.append( sdf['comp7_rate'].mean())                           # 
        stat.append( sdf['comp7_inv'].mean())                            # 
        stat.append( sdf['comp7_rate_percent_diff'].mean())              # 
        stat.append( sdf['comp8_rate'].mean())                           # 
        stat.append( sdf['comp8_inv'].mean())                            # 
        stat.append( sdf['comp8_rate_percent_diff'].mean())              #
        stat.append( sdf['click_bool'].max())                            # clicked on property (ONLY IN TRAIN SET)
        stat.append( sdf['gross_bookings_usd'].iloc[0])                  # Total value of the transaction (ONLY IN TRAIN SET)
        stat.append( sdf['booking_bool'].max())                          # booked the hotel? (ONLY IN TRAIN SET)
       
        stat_col3 = ['comp1_rate', 'comp1_inv', 'comp1_rate_percent_diff',
            'comp2_rate', 'comp2_inv', 'comp2_rate_percent_diff',
            'comp3_rate', 'comp3_inv', 'comp3_rate_percent_diff',
            'comp4_rate', 'comp4_inv', 'comp4_rate_percent_diff',
            'comp5_rate', 'comp5_inv', 'comp5_rate_percent_diff',
            'comp6_rate', 'comp6_inv', 'comp6_rate_percent_diff',
            'comp7_rate', 'comp7_inv', 'comp7_rate_percent_diff',
            'comp8_rate', 'comp8_inv', 'comp8_rate_percent_diff',
            'click_bool', 'gross_bookings_usd', 'booking_bool']
        
        search_ids.append(pd.DataFrame(stat, index = stat_col1 + stat_col2 + stat_col3))
    
    return pd.concat(search_ids,axis = 1).T


# function read in data and process chunks to combine afterwords
def make_data(filename, chunksize):
    
    new_data = []
    
    for df in pd.read_csv(filename, chunksize=chunksize):
        
        new_data.append(process(df))

    result = pd.concat(new_data, axis = 0)
    
    return result
    

