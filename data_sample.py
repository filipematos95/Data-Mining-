import pandas as pd
import random 
import numpy as np
import sys


filename = "training_set_VU_DM_2014.csv"
samples = 1000

if len(sys.argv) > 1:
	samples = int(sys.argv[1])
	if len(sys.argv) > 2:
		filename = sys.argv[2]
		
else: 
	print("Please spicify the size of the sample you want")
	print("Using default 1000")

df = pd.read_csv(filename, nrows = samples) 
df.to_csv(filename[:-4]+"_"+str(samples)+".csv")
