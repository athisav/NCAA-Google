import numpy as np  #linear algebra
import pandas as pd  #CSV read

#seed information
seeds = pd.read_csv('input/WNCAATourneySeeds.csv')
#tour information
tour = pd.read_csv('input/WNCAATourneyCompactResults.csv')

seeds['seed_int'] = seeds['Seed'].apply( lambda x : int(x[1:3]) )
