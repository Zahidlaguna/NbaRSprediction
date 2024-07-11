# %%
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LinearRegression

# %%
nba_data = pd.read_csv('/Users/zlaguna/Downloads/NBA-dataset-stats-player-team-main/team/team_stats_advanced_rs.csv')
nba_data.head()

# %%
print(nba_data.shape)

# %%
nba_data.info()

# %%
nba_data.describe()

# %%
sns.pairplot(nba_data)

# %%
sns.displot(nba_data['W_PCT'], kde=True)
sns.displot(nba_data['E_OFF_RATING'], kde=True)
sns.displot(nba_data['E_DEF_RATING'], kde=True)
sns.displot(nba_data['E_NET_RATING'], kde=True)
sns.displot(nba_data['E_PACE'], kde=True)

# %%
plt.figure(figsize=(50, 50))
plt.title('Offensive efficiency to win percentage')
sns.scatterplot(x='W_PCT', y='E_OFF_RATING', data=nba_data)
for i in range(nba_data.shape[0]):
    plt.text(nba_data.W_PCT[i], nba_data.E_OFF_RATING[i], nba_data.TEAM_NAME[i])

# %%
plt.figure(figsize=(50, 50))
plt.title('Defensive efficiency to win percentage')
sns.scatterplot(x='W_PCT', y='E_DEF_RATING', data=nba_data)
for i in range(nba_data.shape[0]):
    plt.text(nba_data.W_PCT[i], nba_data.E_OFF_RATING[i], nba_data.TEAM_NAME[i])

# %%
plt.figure(figsize=(50, 50))
plt.title('Net rating to win percentage')
sns.scatterplot(x='W_PCT', y='E_NET_RATING', data=nba_data)
for i in range(nba_data.shape[0]):
    plt.text(nba_data.W_PCT[i], nba_data.E_OFF_RATING[i], nba_data.TEAM_NAME[i])

# %%
plt.figure(figsize=(50, 50))
plt.title('Pace to win percentage')
sns.scatterplot(x='W_PCT', y='E_PACE', data=nba_data)
for i in range(nba_data.shape[0]):
    plt.text(nba_data.W_PCT[i], nba_data.E_OFF_RATING[i], nba_data.TEAM_NAME[i])

# %%
plt.figure(figsize=(30,20))
plt.title('Offensive efficiency per Season')
sns.lineplot(x='SEASON', y='E_OFF_RATING', data=nba_data)

# %%
plt.figure(figsize=(30,20))
plt.title('Defensive efficiency per Season')
sns.lineplot(x='SEASON', y='E_DEF_RATING', data=nba_data)

# %%
offensive_wins_corr = nba_data['W_PCT'].corr(nba_data['E_OFF_RATING'])
print("The correlation between offensive efficiency and win percentage is:",offensive_wins_corr * 100, "%")

# %%
defensive_wins_corr = nba_data['W_PCT'].corr(nba_data['E_DEF_RATING'])
print("The correlation between defensive efficiency and win percentage is:",defensive_wins_corr * 100, "%")

# %%
net_wins_corr = nba_data['W_PCT'].corr(nba_data['E_NET_RATING'])
print("The correlation between net rating and win percentage is:",net_wins_corr * 100, "%")

# %%
pace_wins_corr = nba_data['W_PCT'].corr(nba_data['E_PACE'])
print("The correlation between pace and win percentage is:",pace_wins_corr * 100, "%")

# %%
X = nba_data[['E_OFF_RATING', 'E_DEF_RATING', 'E_NET_RATING', 'E_PACE']]
y = nba_data['W_PCT']

# %%
print(X.shape)
print(y.shape)

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

# %%
lrmodel = LinearRegression()
lrmodel.fit(X_train, y_train)

# %%
win_percentage_predictions = lrmodel.predict(X_test)
print(win_percentage_predictions)

# %%
for i in range(len(win_percentage_predictions)):
    print("Predicted win percentage:", win_percentage_predictions[i], "Actual win percentage:", y_test.iloc[i])

# %%
#predicting the win percentage of the 2024-2025 season for each team
teams = nba_data['TEAM_NAME']
offensive_rating = nba_data['E_OFF_RATING']
defensive_rating = nba_data['E_DEF_RATING']
net_rating = nba_data['E_NET_RATING']
pace = nba_data['E_PACE']

team_data = pd.DataFrame({'TEAM_NAME': teams, 'E_OFF_RATING': offensive_rating, 'E_DEF_RATING': defensive_rating, 'E_NET_RATING': net_rating, 'E_PACE': pace})
team_data = team_data.drop_duplicates(subset=['TEAM_NAME'])
team_data = team_data.dropna()
team_data = team_data.reset_index(drop=True)

team_data['W_PCT'] = lrmodel.predict(team_data[['E_OFF_RATING', 'E_DEF_RATING', 'E_NET_RATING', 'E_PACE']])
print(team_data)


# %%
print("The predicted win percentage for the 2024-2025 season for each team is:")
for i in range(team_data.shape[0]):
    print(team_data.TEAM_NAME[i], ":", team_data.W_PCT[i])

# %%
plt.figure(figsize=(50, 50))
plt.title('Predicted win percentage for the 2024-2025 season')
sns.barplot(x='W_PCT', y='TEAM_NAME', data=team_data)

# %%
#neural network model to predict win percentage for the 2024-2025 season
from sklearn.neural_network import MLPRegressor

model = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=1000)
model.fit(X_train, y_train)

win_percentage_predictions = model.predict(X_test)
print(win_percentage_predictions)


# %%
for i in range(len(win_percentage_predictions)):
    print("Predicted win percentage:", win_percentage_predictions[i], "Actual win percentage:", y_test.iloc[i])


# %%
team_data['W_PCT'] = model.predict(team_data[['E_OFF_RATING', 'E_DEF_RATING', 'E_NET_RATING', 'E_PACE']])
print(team_data)

# %%
print("The predicted win percentage for the 2024-2025 season for each team is:")
for i in range(team_data.shape[0]):
    print(team_data.TEAM_NAME[i], ":", team_data.W_PCT[i])

# %%
score = lrmodel.score(X_test, y_test)
print(f'the R2 score of this linear regression model is: {score * 100}%')

# %%
neural_net_score = model.score(X_test, y_test)
print(f'the score of this linear regression model using MLP is: {neural_net_score * 100}%')

# %%
import xgboost as xg 
xgboost = xg.XGBRegressor(objective='reg:squarederror', n_estimators = 10, seed = 123)
xgboost.fit(X_train, y_train)
xgboost_prediction = xgboost.predict(X_test)
xgboost_prediction

# %%
from sklearn.metrics import mean_squared_error as MSE
rmse = np.sqrt(MSE(y_test, xgboost_prediction)) 
print("RMSE : % f" %(rmse))

# %%
import pickle 

with open('xgboost.pkl','wb') as f:
    pickle.dump(xgboost,f)

# %%
import bentoml
bento_model = bentoml.sklearn.save_model('LinearRegression',lrmodel)

# %%
import bentoml
bento_model = bentoml.sklearn.save_model('sklearn_mlp',model)

# %%



