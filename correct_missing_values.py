import pandas as pd
import random 
import numpy as np
import matplotlib.pyplot as plt

filename = "random_samples_1000.csv"
df = pd.read_csv(filename)
#plt.matshow(df.corr())
#plt.show()

prop_country_id = np.unique(df['prop_country_id'])
prop_location_score2_new = []

#print df['prop_location_score2']
print len(df[np.isnan(df['prop_location_score2'])])

for country in prop_country_id:
	print "Country: " + str(country)
	#print np.mean(df.loc[df['prop_country_id'] == country]['prop_location_score2'])
	country_values = np.array(df[df['prop_country_id'] == country]['prop_location_score2'])
	
	df.at[(df['prop_country_id'] == country) & np.isnan(df['prop_location_score2']),'prop_location_score2'] =  np.nanmean(country_values)
	
	if len(df[(df['prop_country_id'] == country) & np.isnan(df['prop_location_score2'])]) > 0:
		print df[(df['prop_country_id'] == country) & (df['prop_location_score2'] == np.nan)]
	#print "Values: "+ str(np.nanmean(country_values))
	
	#if  np.isnan(np.nanmean(country_values)): 
		#print country_values

df.to_csv(filename[:-4]+'_'+'corrected_loc_score_2'+'.csv')
#print df[np.isnan(df['prop_location_score2'])] 