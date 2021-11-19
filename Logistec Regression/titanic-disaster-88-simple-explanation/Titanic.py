import numpy as np 
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import warnings 
warnings.filterwarnings('ignore')


train_df = pd.read_csv('E://Data Science//Training//Datasets//titanic//train.csv')
test_df = pd.read_csv('E://Data Science//Training//Datasets//titanic//test.csv')
subdf = pd.read_csv('E://Data Science//Training//Datasets//titanic//gender_submission.csv')

sub_df = subdf.drop(['PassengerId'],axis =1)
test_df =pd.concat([test_df,sub_df],axis = 1)
df = pd.concat([train_df,test_df],axis =0)
df.head()


list(df.columns.values)

df = df[['PassengerId','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked','Survived']]

df.shape
df.describe()
df.isnull().sum()
sns.heatmap(df.isnull(),cbar=False, cmap='viridis')

df['Survived'].value_counts()
sns.countplot(df['Survived'])

corr_matrix = df.corr()
cmap = sns.diverging_palette(230, 20, as_cmap=True) 
sns.heatmap(corr_matrix, annot=None ,cmap=cmap)

corr_matrix.nlargest(3, 'Survived')['Survived'].index


lis=['Cabin','Name','Ticket']
df= df.drop(lis ,axis=1)

sns.set_style('whitegrid')
sns.countplot(x=df['Survived'],hue=df['Sex'],data=df)

for i in df[['Age' ,'Pclass' ,'Fare','Parch']] :
    print(i,'&','Survived')
    df.hist(column=i, by='Survived')
    
    

cleaning = df.drop(['Survived'],axis = 1)
Survived = df['Survived']
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric_cols = cleaning.select_dtypes(include=numerics)
numeric_cols = numeric_cols.fillna(numeric_cols.mean())

categorical = ['object']
categorical_cols = cleaning.select_dtypes(include=categorical)
categorical_cols = categorical_cols.fillna('none')
categorical_cols = pd.get_dummies(categorical_cols )
cleaned = pd.concat([numeric_cols,categorical_cols],axis= 1)
df = pd.concat([cleaned,Survived],axis = 1)

test_dfn = df.iloc[ 891 : ,:-1]
test_df = df.iloc[ 891 : ,:-1].values
X = df.iloc[:,:-1].values
y = df['Survived'].values

scl = MinMaxScaler(feature_range = (0, 1))
X = scl.fit_transform(X) 
test_df = scl.fit_transform(test_df) 


X_train ,X_test ,y_train ,y_test = train_test_split(X, y , test_size = 0.3, random_state = 44)

# lgm = LogisticRegression()
# lgm = lgm.fit(X_train,y_train)
# y_tpred = lgm.predict(X_train)
# y_pred = lgm.predict(X_test)
# print('train score :',accuracy_score(y_train ,y_tpred ))
# print('test score :',accuracy_score(y_test , y_pred))
# print('con matrix :',confusion_matrix(y_test, y_pred))
# print('report :',classification_report(y_test, y_pred ))


# rnc = RandomForestClassifier(n_estimators=100, max_depth=4,random_state=0)
# rnc = rnc.fit(X_train,y_train)
# y_tpred = rnc.predict(X_train)
# y_pred = rnc.predict(X_test)
# print('train score :',accuracy_score(y_train ,y_tpred ))
# print('test score :',accuracy_score(y_test , y_pred))
# print('con matrix :',confusion_matrix(y_test, y_pred))
# print('report :',classification_report(y_test, y_pred ))


# gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.5,max_depth=3, random_state=140)
# gbc = gbc.fit(X_train,y_train)
# y_tpred = gbc.predict(X_train)
# y_pred = gbc.predict(X_test)
# print('train score :',accuracy_score(y_train ,y_tpred ))
# print('test score :',accuracy_score(y_test , y_pred))
# print('con matrix :',confusion_matrix(y_test, y_pred))
# print('report :',classification_report(y_test, y_pred ))


# svc = SVC()
# svc = svc.fit(X_train,y_train)
# y_tpred = svc.predict(X_train)
# y_pred = svc.predict(X_test)
# print('train score :',accuracy_score(y_train ,y_tpred))
# print('test score :',accuracy_score(y_test , y_pred))
# print('con matrix :',confusion_matrix(y_test, y_pred))
# print('report :',classification_report(y_test, y_pred ))

v1 = LogisticRegression(solver='lbfgs', multi_class='multinomial',random_state=1 , C = 0.5 , tol = 0.001)
v2 = GradientBoostingClassifier(n_estimators=100, learning_rate=0.5,max_depth=3, random_state=140)
v3 = SVC()
eclf = VotingClassifier(estimators=[('lr', v1), ('rf', v2), ('gnb', v3)],voting='hard')

for clf, label in zip([v1, v2, v3, eclf], ['Logistic Regression', 'Gradient Boosting ', 'SVC ', 'Ensemble ']): 
    scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))


svc = SVC()
svc = svc.fit(X_train,y_train)
y_train_pred = svc.predict(X_train)
y_test_pred = svc.predict(X_test)
print('train score :',accuracy_score(y_train ,y_train_pred))
print('test score :',accuracy_score(y_test , y_test_pred))
print('con matrix :',confusion_matrix(y_test, y_test_pred))
print('report :',classification_report(y_test, y_test_pred ))





y_pred = svc.predict(test_df)
Submission = pd.DataFrame({ 'PassengerId': test_dfn['PassengerId'],
                            'Survived': y_pred })
Submission.to_csv("Submission.csv", index=False)

Submission.shape