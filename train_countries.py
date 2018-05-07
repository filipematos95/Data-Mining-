#########################################
#                                       #
#               Group 24                #
#               (2018)                  #
#           Vrije Universiteit          #
#           group by country            #
#                                       #
#########################################

import pandas as pd
import numpy as np
import copy


"""
File reads in preprocced train data to seperate it by prop country id
"""
train = pd.read_csv('preprocess_total.csv', low_memory=False)
country_group = train.groupby('prop_country_id')

for name, group in country_group:
    
    # orange row
    meta1 = ['d', 'c','d', 'd','d', 'c', 'c', 'd','d', 'c','c', 'c','c', 'c','d', 'd'] # discrete (d), continuous (c), string (s)
    meta2 = ['d', 'c','d', 'c','c', 'c', 'c', 'c','c', 'd','c', 'c','c']    
    meta3 = ['c', 'c','c', 'c','c', 'c', 'c', 'c']
    meta4 = ['c', 'c','c', 'c']
    
    index = pd.DataFrame(meta1 + meta2 + meta3 + meta4, index= group.columns).T
    extra = index.copy()
    extra[extra != np.nan] = np.nan
    extra.iloc[0,2] = 'c'
    result = pd.concat([index, extra, group], axis = 0)

    result.to_csv('train_countries/' + str(name) + '.csv', index =False)
        
        

