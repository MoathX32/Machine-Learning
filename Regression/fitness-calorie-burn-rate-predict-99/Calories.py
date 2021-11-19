'''import my libraries '''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
import warnings 
warnings.filterwarnings('ignore')

'''load training dataset'''
exercise = pd.read_csv('E://Data Science//Training//Datasets//Calories//exercise.csv') 
Calories = pd.read_csv('E://Data Science//Training//Datasets//Calories//calories.csv')
calories = Calories.drop(['User_ID'],axis =1)
df = pd.concat([exercise,calories],axis =1)

df.head()

df['Gender'] = df['Gender'].map({'female':0, 'male':1})

df.head()

df.hist()

df.plot.barh()

df.isnull().sum()

sns.heatmap(df.isnull(),cbar=False, cmap='viridis')


df['Calories'].describe()
sns.distplot(df['Calories'])

corr_matrix = df.corr()
cmap = sns.diverging_palette(230, 20, as_cmap=True) 
sns.heatmap(corr_matrix, annot=True ,cmap=cmap)

# df.plot.hexbin(x=df['Gender'], y=df['Calories'], gridsize=20)

df.corr()['Calories'].abs()

C = corr_matrix.nlargest(4, 'Calories')['Calories'].index
for i in C : 
    var = i
    data = pd.concat([df['Calories'], df[var]], axis=1)
    data.plot.scatter(x=var, y='Calories')
    
N = corr_matrix.nsmallest(4,'Calories')['Calories'].index
for i in N :    
    var = i
    data = pd.concat([ df[var], df['Calories']], axis=1)
    
train =  X = df.iloc[:,:-1]
y = df['Calories']

scl = MinMaxScaler(feature_range = (0, 1))
X = scl.fit_transform(X)

X_train ,X_test ,y_train ,y_test = train_test_split(X, y , test_size = 0.3, random_state = 4)

g = GradientBoostingRegressor(n_estimators = 150, learning_rate = 1.5, max_depth = 3)
g = g.fit(X_train,y_train)
score = g.score(X_train,y_train)
percentage = "{:.0%}".format(score)
y_tpred = g.predict(X_test)
print('MAE:', metrics.mean_absolute_error(y_test, y_tpred))  
print('MSE:', metrics.mean_squared_error(y_test, y_tpred))  
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_tpred)))
print('VarScore:',metrics.explained_variance_score(y_test,y_tpred))
print('Acc_Score:',percentage)

# fig, ax = plt.subplots(figsize=(30,10))
# ax.plot(range(len(y_test)), y_test, '-b',label='Actual')
# ax.plot(range(len(y_tpred)), y_tpred, 'r', label='Predicted')
# fig = plt.figure(figsize=(10,5))
# plt.scatter(y_test,y_tpred) 
# plt.plot(y_test,y_test,'r')
# plt.show()


y_pred = g.predict(X)
Submission = pd.DataFrame({ 'User_ID': Calories['User_ID'],
                            'Calories': y_pred })
Submission.to_csv("Submission.csv", index=False)
Submission = Submission.iloc[ :4001 ,:]
Submission.rename(columns={'User_ID': 'id'}, inplace=True)
Submission.shape