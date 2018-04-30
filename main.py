#########################################
#                                       #
#               Group 24                #
#               (2018)                  #
#           Vrije Universiteit          #
#           data exploration            #
#                                       #
#########################################

from preprocess import *

import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline


"""
main.

"""


###################################### example raw plotting ########################################


# read in part of the data and plot
df = pd.read_csv("random_samples_1000.csv")

if plot == True:
  df[df.columns[0:15]].iloc[0:200].plot(subplots= True, figsize = (10,20))
  df[df.columns[15:30]].iloc[0:200].plot(subplots= True, figsize = (10,20))
  df[df.columns[30:45]].iloc[0:200].plot(subplots= True, figsize = (10,20))
  df[df.columns[45:60]].iloc[0:200].plot(subplots= True, figsize = (10,20))

  df[df.columns[0:15]].iloc[2000:2200].plot(subplots= True, figsize = (10,20))
  df[df.columns[15:30]].iloc[0:2000].plot(subplots= True, figsize = (10,20))
  df[df.columns[30:45]].iloc[0:2000].plot(subplots= True, figsize = (10,20))
  df[df.columns[45:60]].iloc[0:2000].plot(subplots= True, figsize = (10,20))


  grr = pd.plotting.scatter_matrix(df[df.columns[0:15]].iloc[:,2:], figsize=(15, 15), marker='.',
      hist_kwds={'bins': 20}, s=20, alpha=.8)

 	
  print(df.describe())

###################################### example preprocessed plotting ########################################


# process the data to search ids (takes a while to run!(1 hour)
filename = "random_samples_1000.csv"
new = make_data(filename, chunksize)
new.to_csv('preprocessed1.csv')

# this goes faster
df = pd.read_csv("new.csv") 
df.index = df['srch_id']
test = df.iloc[:,2:]

# hist
plt.style.use('ggplot')
test.boxplot(figsize=(10,10), rot = 90)
test[test<10000].boxplot(figsize=(10,10), rot = 90)
test[test<1000].boxplot(figsize=(10,10), rot = 90)
test[test<100].boxplot(figsize=(10,10), rot = 90)
test[(test<10) & (test > 0)].boxplot(figsize=(10,10), rot = 90)
test[(test<1) & (test > 0)].boxplot(figsize=(10,10), rot = 90)

# twice as much search id did booking (61409 vs 138390)
plt.hist(test['booking_bool'],  bins=3)
plt.title('booked')
plt.savefig('hist/booked.png')
plt.close()

# plot for each feature a histogram 
groups = test.groupby('booking_bool')
for x in test.columns:
    plt.hist( [groups.get_group(0)[x].dropna(), groups.get_group(1).iloc[0:len(groups.get_group(0))][x].dropna() ],  color=['b','r'], alpha=0.5, bins=10)
    plt.title(x)
    plt.savefig('hist/'+ str(x) + '.png')
    plt.close()

# feature click bool is corrected (need to rerun: was mean instead of max())


columns = ['visitor_location_country_id',
       'visitor_hist_starrating', 'visitor_hist_adr_usd', 'prop_country_id',
       'prop_id', 'prop_starrating', 'prop_review_score', 'prop_brand_bool',
       'prop_location_score1', 'prop_location_score2',
       'prop_log_historical_price', 'position', 'price_usd', 'promotion_flag',
       'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window',
       'srch_adults_count', 'srch_children_count', 'srch_room_count',
       'srch_saturday_night_bool', 'srch_query_affinity_score',
       'orig_destination_distance', 'random_bool', 'comp1_rate', 'comp1_inv',
       'comp1_rate_percent_diff', 'comp2_rate', 'comp2_inv',
       'comp2_rate_percent_diff', 'comp3_rate', 'comp3_inv',
       'comp3_rate_percent_diff', 'comp4_rate', 'comp4_inv',
       'comp4_rate_percent_diff', 'comp5_rate', 'comp5_inv',
       'comp5_rate_percent_diff', 'comp6_rate', 'comp6_inv',
       'comp6_rate_percent_diff', 'comp7_rate', 'comp7_inv',
       'comp7_rate_percent_diff', 'comp8_rate', 'comp8_inv',
       'comp8_rate_percent_diff', 'click_bool', 'gross_bookings_usd',
       'booking_bool']

# pair plot against each other (1000 plots) (WARNING)
for p in itertools.combinations(columns, 2):
    col1 = test[p[0]]
    col2 = test[p[1]]
    
    #plt.scatter(col1,col2, c = test['booking_bool'], alpha=0.2, marker = ".")
    sns.lmplot(x = col1, y = col2, hue = test['booking_bool'])
    
    plt.title(str(p))
    plt.xlabel(p[0])
    plt.ylabel(p[1])
    plt.savefig('scatter/'+ str(p) + '.png')
    plt.close()
    break

#try
test.iloc[:,0:10].data.hist()

x = test.columns[3]
y = test.columns[4]
plt.scatter(test[x].dropna(),test[y].dropna(), c = test['booking_bool'])
plt.title(x)
plt.savefig('scatter/'+ str(test) + '.png')
plt.close()


test
df.head()
len(df.columns)


new = 
new.index = new.iloc[:,0]
new2 = new2.drop('srch_id', 1)

new2 = new2.drop('Unnamed', 1)
new2.head()

len(new2.columns)/4
subsets = [new2.columns[0:13],new2.columns[13:26],new2.columns[26:39],new2.columns[0:52]]
pairs = 

for p in itertools.combinations(subsets, 2):
    grr = pd.plotting.scatter_matrix(new.iloc[:,1:10], figsize=(15, 15), marker='.',
    hist_kwds={'bins': 20}, s=10, alpha=.8)


# plotting first 12 features against each other
grr = pd.plotting.scatter_matrix(new.iloc[:,1:10], figsize=(15, 15), marker='.',
    hist_kwds={'bins': 20}, s=10, alpha=.8)


plt.scatter(new2[new2.columns[3]],new2[new2.columns[4], c = new2['booking_bool'])



# just give a list of age of male/female and corresponding color here
plt.hist([[a for a, s in zip(age, sex) if s=='M'], 
          [a for a, s in zip(age, sex) if s=='F']], 
         color=['b','r'], alpha=0.5, bins=10)
plt.show()

len(new.columns)
new.head()
len(new)
len(new['srch_id'].unique())

# faster (preprocessing allready done)
df = pd.read_csv("new.csv") 
df = df.reset_index()
df.head()

###################################### improvements ########################################



# some featureshave zero if no value is available and averaging these will distort the signal!
# proposed solution -> deal with missing values
# this holds for:
#   prop_review_score
#   prop_brand_bool
#   prop_log_historical_price
#   srch_query_affinity_score
#   orig_destination_distance

# some features areonly in the trainset!
# this holds for:
#   position
#   click_bool
#   gross_bookings_usd
#   booking_bool

# variance in hotel prices is very high due to a variety
# proposed solution -> maketransformation oforiginal data by considering country, fees,..., etc..
# this holds for:
#   price_usd
#   gross_bookings_usd

# performance of the process fuction suffers probably by the creation of all these dataframes
# at each new search id. Solution -> collect first all data in list of list and finally perform:
# df = DataFrame(table, columns=headers)

# there should be a check if a search id is cut in half and if so should pasted together again.



###################################### some info original data set ########################################




columns = ['srch_id', 'date_time', 'site_id', 'visitor_location_country_id',
       'visitor_hist_starrating', 'visitor_hist_adr_usd', 'prop_country_id',
       'prop_id', 'prop_starrating', 'prop_review_score', 'prop_brand_bool',
       'prop_location_score1', 'prop_location_score2',
       'prop_log_historical_price', 'position', 'price_usd', 'promotion_flag',
       'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window',
       'srch_adults_count', 'srch_children_count', 'srch_room_count',
       'srch_saturday_night_bool', 'srch_query_affinity_score',
       'orig_destination_distance', 'random_bool', 'comp1_rate', 'comp1_inv',
       'comp1_rate_percent_diff', 'comp2_rate', 'comp2_inv',
       'comp2_rate_percent_diff', 'comp3_rate', 'comp3_inv',
       'comp3_rate_percent_diff', 'comp4_rate', 'comp4_inv',
       'comp4_rate_percent_diff', 'comp5_rate', 'comp5_inv',
       'comp5_rate_percent_diff', 'comp6_rate', 'comp6_inv',
       'comp6_rate_percent_diff', 'comp7_rate', 'comp7_inv',
       'comp7_rate_percent_diff', 'comp8_rate', 'comp8_inv',
       'comp8_rate_percent_diff', 'click_bool', 'gross_bookings_usd',
       'booking_bool']




