import pandas as pd
import random 
import numpy as np
import matplotlib.pyplot as plt
import sys
import copy as copy


if len(sys.argv) > 1:
	filename = sys.argv[1]
else: 
	filename = "random_samples_1000.csv"


df = pd.read_csv(filename, low_memory=False)


# plt.hist(df['booked'].dropna())
# plt.show()
# plt.hist(df['booked_&_clicked'].dropna())
# plt.show()
# plt.hist(df['clicked'].dropna())
# plt.show()
# plt.hist(df['random_bool'].dropna())
# plt.show()

grouped = df.iloc[2:,:].groupby('booked')
min_group = min(len(grouped.get_group('1.0')),len(grouped.get_group('0.0')) )

booked = grouped.get_group('0.0').sample(n = min_group)
nonbooked = grouped.get_group('1.0').sample(n = min_group)

total = pd.concat([booked, nonbooked], axis = 0)
total = total.sample(n = len(total))
index = df.iloc[0:2,:].copy()

train = total[:int(0.9*min_group)]
test = total[int(-0.1*min_group):]
test = pd.concat([index, test], axis = 0)
train = pd.concat([index, train], axis = 0)

train.to_csv(filename[:-4]+'_'+'train'+'.csv', index = False)
test.to_csv(filename[:-4]+'_'+'test'+'.csv', index= False)