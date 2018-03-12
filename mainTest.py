import numpy as np  #linear algebra
import pandas as pd  #CSV read
import tensorflow

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

##### Load and Merging Dataset #####

#seed information
df_seeds = pd.read_csv('input/WNCAATourneySeeds.csv')
#tour information
df_tour = pd.read_csv('input/WNCAATourneyCompactResults.csv')

df_seeds['seed_int'] = df_seeds['Seed'].apply( lambda x : int(x[1:3]))

df_winseeds = df_seeds.loc[:, ['TeamID', 'Season', 'seed_int']].rename(columns={'TeamID':'WTeamID', 'seed_int':'WSeed'})
df_lossseeds = df_seeds.loc[:, ['TeamID', 'Season', 'seed_int']].rename(columns={'TeamID':'LTeamID', 'seed_int':'LSeed'})
df_dummy = pd.merge(left=df_tour, right=df_winseeds, how='left', on=['Season', 'WTeamID'])
df_concat = pd.merge(left=df_dummy, right=df_lossseeds, on=['Season', 'LTeamID'])

df_concat['DiffSeed'] = df_concat[['LSeed', 'WSeed']].apply(lambda x : 0 if x[0] == x[1] else 1, axis = 1)

df_sample_sub = pd.read_csv('input/WSampleSubmissionStage1.csv')
df_sample_sub['Season'] = df_sample_sub['ID'].apply(lambda x : int(x.split('_')[0]) )
df_sample_sub['TeamID1'] = df_sample_sub['ID'].apply(lambda x : int(x.split('_')[1]) )
df_sample_sub['TeamID2'] = df_sample_sub['ID'].apply(lambda x : int(x.split('_')[2]) )

##### Training Data #####

winners = df_concat.rename( columns = { 'WTeamID' : 'TeamID1', 'LTeamID' : 'TeamID2', 'WScore' : 'Team1_Score','LScore' : 'Team2_Score'}).drop(['WSeed', 'LSeed', 'WLoc'], axis = 1)
winners['Result'] = 1.0

losers = df_concat.rename( columns = { 'WTeamID' : 'TeamID2',  'LTeamID' : 'TeamID1', 'WScore' : 'Team2_Score', 'LScore' : 'Team1_Score'}).drop(['WSeed', 'LSeed', 'WLoc'], axis = 1)
losers['Result'] = 0.0

train = pd.concat( [winners, losers], axis = 0).reset_index(drop = True)

train['Score_Ratio'] = train['Team1_Score'] / train['Team2_Score']
train['Score_Total'] = train['Team1_Score'] + train['Team2_Score']
train['Score_Pct'] = train['Team1_Score'] / train['Score_Total']

train_test_inner = pd.merge( train.loc[ train['Season'].isin([2014, 2015, 2016, 2017]), : ].reset_index(drop = True), df_sample_sub.drop(['ID', 'Pred'], axis = 1), on = ['Season', 'TeamID1', 'TeamID2'], how = 'inner')

team1d_num_ot = train_test_inner.groupby(['Season', 'TeamID1'])['NumOT'].median().reset_index()\
.set_index('Season').rename(columns = {'NumOT' : 'NumOT1'})
team2d_num_ot = train_test_inner.groupby(['Season', 'TeamID2'])['NumOT'].median().reset_index()\
.set_index('Season').rename(columns = {'NumOT' : 'NumOT2'})

num_ot = team1d_num_ot.join(team2d_num_ot).reset_index()

num_ot['NumOT'] = num_ot[['NumOT1', 'NumOT2']].apply(lambda x : round( x.sum() ), axis = 1 )

team1d_score_spread = train_test_inner.groupby(['Season', 'TeamID1'])[['Score_Ratio', 'Score_Pct']].median().reset_index()\
.set_index('Season').rename(columns = {'Score_Ratio' : 'Score_Ratio1', 'Score_Pct' : 'Score_Pct1'})
team2d_score_spread = train_test_inner.groupby(['Season', 'TeamID2'])[['Score_Ratio', 'Score_Pct']].median().reset_index()\
.set_index('Season').rename(columns = {'Score_Ratio' : 'Score_Ratio2', 'Score_Pct' : 'Score_Pct2'})

score_spread = team1d_score_spread.join(team2d_score_spread).reset_index()

#geometric mean of score ratio of team 1 and inverse of team 2
score_spread['Score_Ratio'] = score_spread[['Score_Ratio1', 'Score_Ratio2']].apply(lambda x : ( x[0] * ( x[1] ** -1.0) ), axis = 1 ) ** 0.5

#harmonic mean of score pct
score_spread['Score_Pct'] = score_spread[['Score_Pct1', 'Score_Pct2']].apply(lambda x : 0.5*( x[0] ** -1.0 ) + 0.5*( 1.0 - x[1] ) ** -1.0, axis = 1 ) ** -1.0

X_train = train_test_inner.loc[:, ['Season', 'NumOT', 'Score_Ratio', 'Score_Pct']]
train_labels = train_test_inner['Result']

train_test_outer = pd.merge( train.loc[ train['Season'].isin([2014, 2015, 2016, 2017]), : ].reset_index(drop = True), df_sample_sub.drop(['ID', 'Pred'], axis = 1), on = ['Season', 'TeamID1', 'TeamID2'], how = 'outer' )

train_test_outer = train_test_outer.loc[train_test_outer['Result'].isnull(),  ['TeamID1', 'TeamID2', 'Season']]

train_test_missing = pd.merge( pd.merge( score_spread.loc[:, ['TeamID1', 'TeamID2', 'Season', 'Score_Ratio', 'Score_Pct']],train_test_outer, on = ['TeamID1', 'TeamID2', 'Season']),
num_ot.loc[:, ['TeamID1', 'TeamID2', 'Season', 'NumOT']], on = ['TeamID1', 'TeamID2', 'Season'])

#scale data for keras
X_test = train_test_missing.loc[:, ['Season', 'NumOT', 'Score_Ratio', 'Score_Pct']]

n = X_train.shape[0]

train_test_merge = pd.concat( [X_train, X_test], axis = 0 ).reset_index(drop = True)

train_test_merge = pd.concat( [pd.get_dummies( train_test_merge['Season'].astype(object) ), train_test_merge.drop('Season', axis = 1) ], axis = 1 )

train_test_merge = pd.concat( [pd.get_dummies( train_test_merge['NumOT'].astype(object) ),train_test_merge.drop('NumOT', axis = 1) ], axis = 1 )

X_train = train_test_merge.loc[:(n - 1), :].reset_index(drop = True)
X_test = train_test_merge.loc[n:, :].reset_index(drop = True)

x_max = X_train.max()
x_min = X_train.min()

X_train = ( X_train - x_min ) / ( x_max - x_min + 1e-14)
X_test = ( X_test - x_min ) / ( x_max - x_min + 1e-14)

##### Generate #####

seed = 8
np.random.seed(seed)

# create model
model = Sequential()
model.add(Dense(80, input_dim=8, kernel_initializer='lecun_normal', activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(60, kernel_initializer='lecun_normal', activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(50, kernel_initializer='lecun_normal', activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(45, kernel_initializer='lecun_normal', activation='relu'))
model.add(Dropout(0.09))
model.add(Dense(30, kernel_initializer='lecun_normal', activation='relu'))
model.add(Dropout(0.12))
model.add(Dense(15, kernel_initializer='lecun_normal', activation='relu'))
model.add(Dropout(0.09))
model.add(Dense(5, kernel_initializer='lecun_normal', activation='softmax'))
model.add(Dropout(0.09))
model.add(Dense(1, kernel_initializer='lecun_normal', activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.fit(X_train, train_labels, epochs=6000)

##### Store #####

train_test_inner['Pred1'] = model.predict_proba(X_train)
train_test_missing['Pred1'] = model.predict_proba(X_test)

sub = pd.merge(df_sample_sub, pd.concat( [train_test_missing.loc[:, ['Season', 'TeamID1', 'TeamID2', 'Pred1']], train_test_inner.loc[:, ['Season', 'TeamID1', 'TeamID2', 'Pred1']] ], axis = 0).reset_index(drop = True), on = ['Season', 'TeamID1', 'TeamID2'], how = 'outer')

team1_probs = sub.groupby('TeamID1')['Pred1'].apply(lambda x : (x ** -1.0).mean() ** -1.0 ).fillna(0.5).to_dict()
team2_probs = sub.groupby('TeamID2')['Pred1'].apply(lambda x : (x ** -1.0).mean() ** -1.0 ).fillna(0.5).to_dict()

sub['Pred'] = sub[['TeamID1', 'TeamID2','Pred1']]\
.apply(lambda x : team1_probs.get(x[0]) * ( 1 - team2_probs.get(x[1]) ) if np.isnan(x[2]) else x[2], axis = 1)

sub[['ID', 'Pred']].to_csv('sub.csv', index = False)

print(sub[['ID', 'Pred']].head(20))
