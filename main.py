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

x_columns = list(X.columns.values)
x_columns = [x for x in x_columns if x not in ['f_Seed_region','s_Seed_region','winning','Season']]

y = X['winning']
X = X.drop('winning',axis=1)

#Create and train data set
X_train = X[:252]
y_train = y[:252]
X_test = X[252:]
y_test = y[252:]
#Divide test data
t_2014 = X_test[:63]
t_2015 = X_test[63:126]
t_2016 = X_test[126:189]
t_2017 = X_test[189:252]

teams_2014 = sorted(list(set(list(set(t_2014['WTeamID'])) + list(set(t_2014['LTeamID'])))))
teams_2015 = sorted(list(set(list(set(t_2015['WTeamID'])) + list(set(t_2015['LTeamID'])))))
teams_2016 = sorted(list(set(list(set(t_2016['WTeamID'])) + list(set(t_2016['LTeamID'])))))
teams_2017 = sorted(list(set(list(set(t_2017['WTeamID'])) + list(set(t_2017['LTeamID'])))))

teams_list = [teams_2014,teams_2015,teams_2016,teams_2017]

row_num = int(((64*63)/2)*4)
col_num = 2

sub_df = pd.DataFrame(np.zeros((row_num,col_num)),columns=['FirstID','SecondID'])

for i in range(0,row_num,2016):
    l = 0 + i
    u = 63 + i
    r = 0
    m = 63
    k = teams_list[int(i/2016)]
    while(0 < m):
        for j in range(l,u):
            sub_df['FirstID'][j] = k[r]
        r = r+1
        l = u   
        u = u + m - 1
        m = m - 1
        
for i in range(0,row_num,2016):
    l = 0 + i
    u = 63 + i
    r = 0
    m = 63
    k = teams_list[int(i/2016)]
    while(0 < m):
        t = 0
        for j in range(l,u):
            sub_df['SecondID'][j] = k[r+t+1]
            t = t + 1
        r = r+1
        l = u   
        u = u + m - 1
        m = m - 1   

sub_df['year'] = 0
sub_df['year'][:2016]=2014
sub_df['year'][2016:int(2016*2)]=2015
sub_df['year'][int(2016*2):int(2016*3)]=2016
sub_df['year'][int(2016*3):]=2017

#creating an empty dataframe
testing = pd.DataFrame(np.zeros((len(sub_df),len(x_columns))) ,columns=x_columns)

#Renaming columns
testing = testing.rename(columns={'WTeamID':'FirstID','LTeamID':'SecondID'})

for i in range(len(sub_df)):
    season = int(sub_df.iloc[i]['year'])
    f_t = int(sub_df.iloc[i]['FirstID'])
    s_t = int(sub_df.iloc[i]['SecondID'])
    testing.iloc[i]['FirstID'] = f_t
    testing.iloc[i]['SecondID'] = s_t
    for j in list(testing.columns.values)[:19]:
        k = j[2:]
        testing.iloc[i][j] = df_list[season-2010].loc[f_t,k]
    for j in list(testing.columns.values)[19:38]:
        k = j[2:]
        testing.iloc[i][j] = df_list[season-2010].loc[s_t,k]  
        

#Deleting some columns which we don't put in the model.
del X['f_Seed_region'],X['s_Seed_region'],X['WTeamID'],X['LTeamID'],X['Season']
del testing['FirstID'],testing['SecondID']

X_train = X[:252]
y_train = y[:252]

#Checking number of columns is equal
X_train.shape[1] == testing.shape[1]

def make_suitable_column(abc):
    yr = int(abc['year'])
    fi = int(abc['FirstID'])
    si = int(abc['SecondID'])
    k = str(yr) + '_' + str(fi) + '_' + str(si)
    return str(k)
    
submission_column = sub_df.apply(make_suitable_column,axis=1)
submission_column.name = 'ID'

X_train = X_train.values
y_train = y_train.values

from sklearn.metrics import log_loss

#Fitting and predicting
from sklearn.linear_model import LogisticRegression
model1 = LogisticRegression()
model1.fit(X_train,y_train)
y_pred2 = model1.predict_proba(testing.values)
s_f2 = pd.DataFrame(1 - y_pred2)
s_f2 = s_f2[0]
s_f2.name = 'pred'
submission_df2 = pd.concat([submission_column,s_f2],axis=1)

####################
# Submit #
####################
submission_df2.to_csv('sub.csv',index=False)
