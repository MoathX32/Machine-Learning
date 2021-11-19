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
train_df = pd.read_csv('E://Data Science//Training//Datasets//house-prices-advanced-regression-techniques//train.csv') 
test_df = pd.read_csv('E://Data Science//Training//Datasets//house-prices-advanced-regression-techniques//test.csv')
Sub = pd.read_csv('E://Data Science//Training/Datasets//house-prices-advanced-regression-techniques//sample_submission.csv')
Sub = Sub.drop(['Id'],axis =1)
test_df =pd.concat([test_df,Sub],axis = 1)
df = pd.concat([train_df,test_df],axis =0)
df.head()
df.describe()
df.isnull().sum()
lis=['MiscFeature','Fence','PoolQC','Alley']
df= df.drop(lis ,axis=1)
df.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1, figsize=(12,12))

df['SalePrice'].describe()
sns.distplot(df['SalePrice'])

corr_matrix = df.corr()
corr_mat = df.drop('Id',axis=1).corr()
f, ax = plt.subplots(figsize=(12, 10)) 
cmap = sns.diverging_palette(230, 20, as_cmap=True) 
sns.heatmap(corr_matrix, annot=None ,cmap=cmap)

df.corr()['SalePrice'].abs()

C = corr_matrix.nlargest(5, 'SalePrice')['SalePrice'].index
for i in C : 
    var = i
    data = pd.concat([df['SalePrice'], df[var]], axis=1)
    data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
# df = df.drop(df[df['Id'] == 1299].index)
# df = df.drop(df[df['Id'] == 524].index) 
        

N = corr_mat.nsmallest((25),'SalePrice')['SalePrice'].index
for n in N :
    df = df.drop(n ,axis=1)



cleaning = df.drop(['SalePrice'],axis = 1)
SalePrice = df['SalePrice']

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric_cols = cleaning.select_dtypes(include=numerics)
numeric_cols = numeric_cols.fillna(numeric_cols.mean())



categorical = ['object']
categorical_cols = cleaning.select_dtypes(include=categorical)
categorical_cols = categorical_cols.fillna('none')
categorical_cols = pd.get_dummies(categorical_cols )

cleaned = pd.concat([numeric_cols,categorical_cols],axis= 1)

df = pd.concat([cleaned,SalePrice],axis = 1)
# train_df=train_df.drop([523],axis = 0)
# train_df=train_df.drop([1298],axis = 0)

tst_df = df.iloc[ 1460 : ,:-1]
X = df.iloc[:1460,:-1]
y = df.iloc[:1460,-1]


scl = MinMaxScaler()
X = scl.fit_transform(X)
tst_df = scl.fit_transform(tst_df)

X_train ,X_test ,y_train ,y_test = train_test_split(X, y , test_size = 0.3, random_state = 4)

# LN = LinearRegression()
# LN.fit(X_train,y_train)
# y_pred = LN.predict(X_test)
# LN.score(X_train, y_train)


# SGD =SGDRegressor()
# SGD.fit(X_train,y_train)
# y_pred = SGD.predict(X_test)
# SGD.score(X_train, y_train)


# svr = SVR(gamma='scale', C=0.00000001, epsilon=0.2)
# svr.fit(X_train,y_train)
# y_pred = svr.predict(X_test)
# svr.score(X_train, y_train)
# mean_squared_error(y_train, y_pred)
# mean_absolute_error(y_pred,y_test)

# median_absolute_error(y_test, y_pred)



g = GradientBoostingRegressor(n_estimators = 200, learning_rate = 1.5, max_depth = 3)
train = g.fit(X_train,y_train)
score = g.score(X_train,y_train)
percentage = "{:.0%}".format(score)
y_pred = g.predict(X_test)
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))  
print('MSE:', metrics.mean_squared_error(y_test, y_pred))  
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('VarScore:',metrics.explained_variance_score(y_test,y_pred))
print('Acc_Score:',percentage)


fig, ax = plt.subplots(figsize=(30,10))
ax.plot(range(len(y_test)), y_test, '-b',label='Actual')
ax.plot(range(len(y_pred)), y_pred, 'r', label='Predicted')
fig = plt.figure(figsize=(10,5))
plt.scatter(y_test,y_pred) 
plt.plot(y_test,y_test,'r')
plt.show()


y_predw = train.predict(tst_df)
Submission = pd.DataFrame({ 'Id': test_df['Id'],
                            'SalePrice': y_predw })
Submission.to_csv("Submission.csv", index=False)

Submission.shape