'''import my libraries '''
import numpy as np
import pandas as pd     
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

'''load my data for processing and analysis'''
data = pd.read_csv("E:\\Data Science\\Training\\sonar data.csv",header=None)
data.head(10) 
# =============================================================================
encoder = LabelEncoder()
data[60] = encoder.fit_transform(data[60])
data.head(10)
# =============================================================================
data.info()  # so we don't have missing values
data.describe()
data[60].value_counts()
# =============================================================================
conv = data.corr()
f, ax = plt.subplots(figsize=(12, 10)) 
cmap = sns.diverging_palette(230, 20, as_cmap=True) 
sns.heatmap(conv, annot=None ,cmap=cmap)

data.plot(kind='density', subplots=True, layout=(8,8), sharex=False, legend=False, fontsize=1, figsize=(12,12))
plt.show()

df = pd.DataFrame(data)
c = df.corr().abs()
s = c.unstack()
y_corr = s[60]
del y_corr[60]
y_corr
print(max(y_corr)) #so data[10] has the max corr 0.4328549 with y data[60]


conv.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1, figsize=(12,12))
plt.show()
    
''' split the data'''
X = data.drop(60 ,1)
y = data[60]
X_train,X_test,y_train,y_test = train_test_split(X ,y ,test_size = 0.1, random_state=1)

'''QDA_model'''
model_QDA = QuadraticDiscriminantAnalysis(0.0001)
model_QDA.fit(X_train, y_train)
        
y_train_pred = model_QDA.predict(X_train)
train_accurcy = accuracy_score(y_train_pred,y_train)
print(train_accurcy)
   
y_test_pred = model_QDA.predict(X_test)
test_accurcy = accuracy_score(y_test_pred,y_test)
print(test_accurcy)
                                   
#### print(model_QDA.score(y_test_pred,y_test))  ValueError: could not convert string to float: 'M'
con = confusion_matrix(y_test,y_test_pred)
hmap = sns.heatmap(con,annot=True,fmt="d")
print(hmap)

'''Logistic_Regression_model'''
logistic_model = LogisticRegression(penalty='l2',solver='lbfgs',C=1.0,random_state=1)
logistic_model.fit(X_train, y_train)
        
y_train_pred = logistic_model.predict(X_train)
train_accurcy = accuracy_score(y_train_pred,y_train)
print(train_accurcy)

y_test_pred = logistic_model.predict(X_test)
test_accurcy = accuracy_score(y_test_pred,y_test)
print(test_accurcy)
    
con = confusion_matrix(y_test,y_test_pred)
hmap =sns.heatmap(con,annot=True,fmt="d")
print(hmap)


''''SVC_model'''
SVC_model = SVC()
SVC_model.fit(X_train, y_train)

y_train_pred = SVC_model.predict(X_train)
train_accurcy = accuracy_score(y_train_pred,y_train)
print(train_accurcy)
    
y_test_pred = SVC_model.predict(X_test)
test_accurcy = accuracy_score(y_test_pred,y_test)
print(test_accurcy)
        
con = confusion_matrix(y_test,y_test_pred)
hmap =sns.heatmap(con,annot=True,fmt="d")
print(hmap)

'''Random_Forest_model'''
forest_model = RandomForestClassifier(n_estimators = 50,max_depth=5)
forest_model.fit(X_train, y_train)

y_train_pred = forest_model.predict(X_train)
train_accurcy = accuracy_score(y_train_pred,y_train)
print(train_accurcy)

y_test_pred = forest_model.predict(X_test)
test_accurcy = accuracy_score(y_test_pred,y_test)
print(test_accurcy)
    
con = confusion_matrix(y_test,y_test_pred)
hmap =sns.heatmap(con,annot=True,fmt="d")
print(hmap)
