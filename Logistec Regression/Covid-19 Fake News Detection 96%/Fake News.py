'''import my libraries '''
import numpy as np
import pandas as pd
import re 
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier
import warnings 
warnings.filterwarnings('ignore')


'''load training dataset'''
train_df = pd.read_csv('E://Data Science//Training//Datasets//fake-news//train.csv') 
test_df = pd.read_csv('E://Data Science//Training//Datasets//fake-news//test.csv')
submit = pd.read_csv('E://Data Science//Training//Datasets//fake-news//submit.csv')
submit = submit.drop(['id'],axis =1)
test_df = pd.concat([test_df,submit],axis = 1)
df = pd.concat([train_df,test_df],axis =0)
df = df.drop(['id'],axis =1)
df.label.value_counts()
df.isnull().sum()
df = df.fillna('')
df.isnull().sum()
df['content'] = df['author']+' '+df['title'] +' '+df['text']


import nltk
nltk.download('stopwords')

stem = PorterStemmer()
def stemming(content):
    content = re.sub('[^a-zA-Z]',' ',content)
    content = content.lower()
    content = content.split()
    content = [stem.stem(word) for word in content if not word in stopwords.words('english')]
    content = ' '.join(content)
    return content
df['content'] = df['content'].apply(stemming)

X = df['content'].values
y = df['label'].values

vectorizer = TfidfVectorizer()
vectorizer.fit(X)
X = vectorizer.transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size=0.3, random_state=1)

sgd = SGDClassifier()     
sgd = sgd.fit(X_train,Y_train)
y_tpred = sgd.predict(X_train)
y_pred = sgd.predict(X_test)
print('train score :',accuracy_score(Y_train ,y_tpred ))
print('test score :',accuracy_score(Y_test , y_pred))
print('con matrix :',confusion_matrix(Y_test, y_pred))
print('report :',classification_report(Y_test, y_pred ))