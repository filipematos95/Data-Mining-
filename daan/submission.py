#########################################
#                                       #
#               Group 24                #
#               (2018)                  #
#           Vrije Universiteit          #
#           submission widget           #
#                                       #
#########################################

import numpy as np
import itertools
import pandas as pd
import numpy as np
import sys
import copy as copy
import Orange 

"""
widget that process data to correct submission format
"""

test = in_data.copy()

a = pd.DataFrame(np.array(test), columns = ['srch_id', 'pred', 'prop_id'])
a = a.sort_values(['srch_id', 'pred'],ascending=[True, False]).copy()
a.drop('pred', axis = 1, inplace = True)
out_data = Orange.data.Table(a)
