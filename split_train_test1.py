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

index = df.iloc[0:2,:].copy()

train = df[:int(0.9*len(df))]

test = df[int(-0.1*len(df)):]
test = pd.concat([index, test], axis = 0)

train.to_csv(filename[:-4]+'_'+'train'+'.csv', index = False)
test.to_csv(filename[:-4]+'_'+'test'+'.csv', index= False)