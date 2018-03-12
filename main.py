import numpy as np #linear algebra
import scipy as sci #maths ops
import pandas as pd #CSV read
pd.set_option('display.max_columns', None)

#from sklearn - use scikit
#from matplotlib import pyplot as plt - stats

#######################################
# Load and parse files in from CSV using pandas #
#######################################
teams = pd.read_csv("input/WTeams.csv") #Get Teams

seasons = pd.read_csv("input/WSeasons.csv") #Get Seasons

seed = pd.read_csv("input/WNCAATourneySeeds.csv") #Get Tourney Seeds

seed['region'] = seed['Seed'].apply(lambda x: x[0])
seed['no'] = seed['Seed'].apply(lambda x: x[1:])

rscr = pd.read_csv('input/WRegularSeasonCompactResults.csv')

ntcr = pd.read_csv('input/WNCAATourneyCompactResults.csv')
years = sorted(list(set(ntcr['Season'])))
year_team_dict = {}

for i in years:
    year_team_dict[str(i)] = list(set(list(set(ntcr[ntcr['Season'] ==i]['WTeamID'])) + list(set(ntcr[ntcr['Season'] ==i]['LTeamID']))))


###############
# Train model #
###############

####################
# Make predicition #
####################