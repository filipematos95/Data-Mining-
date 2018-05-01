import pandas as pd
import random 
import numpy as np
import sys

filename = "training_set_VU_DM_2014.csv"

n = sum(1 for line in open(filename)) - 1

if len(sys.argv) > 1:
	s = int(sys.argv[1])
else: 
	print("Please spicify the size of the sample you want")
	print("Using default 1000")

skip = sorted(random.sample(xrange(1,n+1),n-s))

df = pd.read_csv(filename, skiprows=skip)

df.to_csv("random_samples_"+str(s)+".csv")
