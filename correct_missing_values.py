import pandas as pd
import random 
import numpy as np
import matplotlib.pyplot as plt

filename = "random_samples_1000.csv"
df = pd.read_csv(filename)
#plt.matshow(df.corr())
#plt.show()

prop_country_id = np.unique(df['prop_country_id'])

for country in prop_country_id:
	print "Country: " + str(country)
	#print np.mean(df.loc[df['prop_country_id'] == country]['prop_location_score2'])
	country_values = np.array(df['prop_location_score2'][df['prop_country_id'] == country])
	if  np.isnan(np.nanmean(country_values)): 
		print country_values
