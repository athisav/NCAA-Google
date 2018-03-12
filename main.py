import numpy as np  #linear algebra
import scipy as sci  #maths ops
import pandas as pd  #CSV read
pd.set_option('display.max_columns', None)

#from sklearn - use scikit
#from matplotlib import pyplot as plt - stats

#######################################
# Load and parse files in from CSV using pandas #
#######################################
teams = pd.read_csv("input/WTeams.csv")  #Get Teams

seasons = pd.read_csv("input/WSeasons.csv")  #Get Seasons

seed = pd.read_csv("input/WNCAATourneySeeds.csv")  #Get Tourney Seeds

seed['region'] = seed['Seed'].apply(lambda x: x[0])
seed['no'] = seed['Seed'].apply(lambda x: x[1:])

rscr = pd.read_csv('input/WRegularSeasonCompactResults.csv')  #regular compact

ntcr = pd.read_csv('input/WNCAATourneyCompactResults.csv')  #tourney compact
years = sorted(list(set(ntcr['Season'])))
year_team_dict = {}

for i in years:
	year_team_dict[str(i)] = list(
	    set(
	        list(set(ntcr[ntcr['Season'] == i]['WTeamID'])) +
	        list(set(ntcr[ntcr['Season'] == i]['LTeamID']))))

####
rsdr = pd.read_csv('input/WRegularSeasonDetailedResults_PrelimData2018.csv'
                   )  #regular prelims detail
#2 Pointers
rsdr['WFGM2'] = rsdr['WFGM'] - rsdr['WFGM3']
rsdr['WFGA2'] = rsdr['WFGA'] - rsdr['WFGA3']
rsdr['LFGM2'] = rsdr['LFGM'] - rsdr['LFGM3']
rsdr['LFGA2'] = rsdr['LFGA'] - rsdr['LFGA3']

clms = [
    'Score', 'EScore', 'FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA', 'OR', 'DR',
    'Ast', 'TO', 'Stl', 'Blk', 'PF', 'FGM2', 'FGA2'
]

df_2010 = pd.DataFrame(
    np.zeros((len(set(rsdr[rsdr['Season'] == 2010]['WTeamID'])), 17)),
    columns=clms,
    index=list(set(rsdr[rsdr['Season'] == 2010]['WTeamID'])))
df_2011 = pd.DataFrame(
    np.zeros((len(set(rsdr[rsdr['Season'] == 2011]['WTeamID'])), 17)),
    columns=clms,
    index=list(set(rsdr[rsdr['Season'] == 2011]['WTeamID'])))
df_2012 = pd.DataFrame(
    np.zeros((len(set(rsdr[rsdr['Season'] == 2012]['WTeamID'])), 17)),
    columns=clms,
    index=list(set(rsdr[rsdr['Season'] == 2012]['WTeamID'])))
df_2013 = pd.DataFrame(
    np.zeros((len(set(rsdr[rsdr['Season'] == 2013]['WTeamID'])), 17)),
    columns=clms,
    index=list(set(rsdr[rsdr['Season'] == 2013]['WTeamID'])))
df_2014 = pd.DataFrame(
    np.zeros((len(set(rsdr[rsdr['Season'] == 2014]['WTeamID'])), 17)),
    columns=clms,
    index=list(set(rsdr[rsdr['Season'] == 2014]['WTeamID'])))
df_2015 = pd.DataFrame(
    np.zeros((len(set(rsdr[rsdr['Season'] == 2015]['WTeamID'])), 17)),
    columns=clms,
    index=list(set(rsdr[rsdr['Season'] == 2015]['WTeamID'])))
df_2016 = pd.DataFrame(
    np.zeros((len(set(rsdr[rsdr['Season'] == 2016]['WTeamID'])), 17)),
    columns=clms,
    index=list(set(rsdr[rsdr['Season'] == 2016]['WTeamID'])))
df_2017 = pd.DataFrame(
    np.zeros((len(set(rsdr[rsdr['Season'] == 2017]['WTeamID'])), 17)),
    columns=clms,
    index=list(set(rsdr[rsdr['Season'] == 2017]['WTeamID'])))

df_list = [
    df_2010, df_2011, df_2012, df_2013, df_2014, df_2015, df_2016, df_2017
]

df_list[0].shape

year = 2010
for m in df_list:
	for i in list(set(rsdr[rsdr['Season'] == year]['LTeamID'])):
		klm = pd.DataFrame()
		klm = rsdr[(rsdr['Season'] == year) & (
		    (rsdr['WTeamID'] == i) | (rsdr['LTeamID'] == i))]
		for j in clms:
			if j == 'EScore':
				m.loc[i, j] = (
				    klm[klm['WTeamID'] == i]['LScore'].values.sum() +
				    klm[klm['LTeamID'] == i]['WScore'].values.sum()) / len(klm)
			else:
				m.loc[i, j] = (
				    klm[klm['WTeamID'] == i]['W' + j].values.sum() +
				    klm[klm['LTeamID'] == i]['L' + j].values.sum()) / len(klm)
	year = year + 1

#Zero for seed values
df_2010['Seed'],df_2011['Seed'],df_2012['Seed'],df_2013['Seed'],df_2014['Seed'],df_2015['Seed'],df_2016['Seed'],df_2017['Seed']=0,0,0,0,0,0,0,0

year = 2010
r = 0
for i in df_list:
    m = i.loc[pd.Series(seed[seed['Season'] ==year]['TeamID']).sort_values(ascending=True),:]
    for j in list(m.index):
        m.loc[j,'Seed'] = list(seed[(seed['Season'] ==year)&(seed['TeamID'] ==j)]['Seed'])[0]
    df_list[r] = m    
    year = year + 1
    r = r+1

k = 0
for i in df_list:
    i['Seed_no'] = i['Seed'].map(lambda x: int(x[1:3]))
    i['Seed_region'] = i['Seed'].map(lambda x: x[0])
    df_list[k] = i
    k = k+1
    
k = 0
for i in df_list:
    del i['Seed']
    df_list[k] = i
    k = k + 1

for m in df_list:
    m['diff'] = m['Score'] - m['EScore']

###############
# Train model #
###############

####################
# Make predicition #
####################
