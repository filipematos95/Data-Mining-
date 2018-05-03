import pandas as pd
import random 
import numpy as np
import matplotlib.pyplot as plt

filename = "random_samples_500000.csv"
df = pd.read_csv(filename)

search_id = np.unique(df['srch_id'])
print len(df['srch_id'])
print len(search_id)