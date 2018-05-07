import pandas as pd
import random 
import numpy as np
import matplotlib.pyplot as plt
import sys

if len(sys.argv) > 1:
	filename = sys.argv[1]
else: 
	filename = "random_samples_1000.csv"


df = pd.read_csv(filename)

train = df[:int(0.9*len(df))]
test = df[int(-0.1*len(df)):]

train.to_csv(filename[:-4]+'_'+'train'+'.csv')
test.to_csv(filename[:-4]+'_'+'test'+'.csv')