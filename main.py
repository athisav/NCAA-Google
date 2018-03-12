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
rsdr = pd.read_csv('input/WRegularSeasonDetailedResults_PrelimData2018.csv')  #regular prelims detail
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

####
ntdr = pd.read_csv('input/WNCAATourneyDetailedResults_PrelimData2018.csv') # tourney prelims detail

def winning(data):
    if data['WTeamID'] < data['LTeamID']:
        return 1
    else:
        return 0

ntdr['winning'] = ntdr.apply(winning,axis=1)

#2 point throws
ntdr['WFGM2'] = ntdr['WFGM'] - ntdr['WFGM3']
ntdr['WFGA2'] = ntdr['WFGA'] - ntdr['WFGA3']
ntdr['LFGM2'] = ntdr['LFGM'] - ntdr['LFGM3']
ntdr['LFGA2'] = ntdr['LFGA'] - ntdr['LFGA3']

f_columns = ['f_'+str(x) for x in list(df_list[7].columns.values)]
s_columns = ['s_'+str(x) for x in list(df_list[7].columns.values)]
t_columns = f_columns + s_columns

nc_2010 = pd.concat([pd.DataFrame(np.zeros((ntdr[ntdr['Season'] ==2010].shape[0],df_list[0].shape[1]*2)),columns= t_columns),ntdr[ntdr['Season'] ==2010][['WTeamID','LTeamID','winning']].reset_index(drop=True)],axis=1)
nc_2011 = pd.concat([pd.DataFrame(np.zeros((ntdr[ntdr['Season'] ==2011].shape[0],df_list[0].shape[1]*2)),columns= t_columns),ntdr[ntdr['Season'] ==2011][['WTeamID','LTeamID','winning']].reset_index(drop=True)],axis=1)
nc_2012 = pd.concat([pd.DataFrame(np.zeros((ntdr[ntdr['Season'] ==2012].shape[0],df_list[0].shape[1]*2)),columns= t_columns),ntdr[ntdr['Season'] ==2012][['WTeamID','LTeamID','winning']].reset_index(drop=True)],axis=1)
nc_2013 = pd.concat([pd.DataFrame(np.zeros((ntdr[ntdr['Season'] ==2013].shape[0],df_list[0].shape[1]*2)),columns= t_columns),ntdr[ntdr['Season'] ==2013][['WTeamID','LTeamID','winning']].reset_index(drop=True)],axis=1)
nc_2014 = pd.concat([pd.DataFrame(np.zeros((ntdr[ntdr['Season'] ==2014].shape[0],df_list[0].shape[1]*2)),columns= t_columns),ntdr[ntdr['Season'] ==2014][['WTeamID','LTeamID','winning']].reset_index(drop=True)],axis=1)
nc_2015 = pd.concat([pd.DataFrame(np.zeros((ntdr[ntdr['Season'] ==2015].shape[0],df_list[0].shape[1]*2)),columns= t_columns),ntdr[ntdr['Season'] ==2015][['WTeamID','LTeamID','winning']].reset_index(drop=True)],axis=1)
nc_2016 = pd.concat([pd.DataFrame(np.zeros((ntdr[ntdr['Season'] ==2016].shape[0],df_list[0].shape[1]*2)),columns= t_columns),ntdr[ntdr['Season'] ==2016][['WTeamID','LTeamID','winning']].reset_index(drop=True)],axis=1)
nc_2017 = pd.concat([pd.DataFrame(np.zeros((ntdr[ntdr['Season'] ==2017].shape[0],df_list[0].shape[1]*2)),columns= t_columns),ntdr[ntdr['Season'] ==2017][['WTeamID','LTeamID','winning']].reset_index(drop=True)],axis=1)

nc_list = [nc_2010,nc_2011,nc_2012,nc_2013,nc_2014,nc_2015,nc_2016,nc_2017]

r = 0
for i in nc_list:
    for j in range(len(i)):
        for m in range(len(f_columns)):
            i.iloc[j,m] = df_list[r].loc[min(i.loc[j,'LTeamID'],i.loc[j,'WTeamID']),f_columns[m][2:]]
        for m in range(len(f_columns),len(f_columns)*2):
            i.iloc[j,m] = df_list[r].loc[max(i.loc[j,'LTeamID'],i.loc[j,'WTeamID']),s_columns[m-len(f_columns)][2:]]
    nc_list[r] = i
    r = r+1

#Seasons to rows
j=2010
for i in nc_list:
    i['Season'] = j
    j = j + 1
###############
# Train model #
###############
X = pd.concat([nc_list[0],nc_list[1],nc_list[2],nc_list[3],nc_list[4],nc_list[5],nc_list[6],nc_list[7]])
X = X.reset_index(drop=True)


####################
# Submit #
####################
