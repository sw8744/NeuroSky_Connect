#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def load_data(idlePath:str,sleepPath:str,sleepyPath:str):
    df1 = pd.read_csv(idlePath).assign(result=lambda x:'IDLE')
    df2 = pd.read_csv(sleepPath).assign(result=lambda x:'SLEEP')
    df3 = pd.read_csv(sleepyPath).assign(result=lambda x:'SLEEPLY')

    sampleCnt = min(len(df1),len(df2),len(df3))
    # df1 = df1.sample(sampleCnt)
    # df2 = df2.sample(sampleCnt)
    # df3 = df3.sample(sampleCnt)

    result = pd.concat([df1,df2,df3])
    return result

df = load_data('result_idle.csv','result_sleep_1.csv','result_sleepy_1.csv') # 데이터 불러오는 코드임.

df

Y = df['result'].to_numpy()
X = df.drop(columns=['result']).to_numpy()

rescale = StandardScaler()
rescale.fit(X)
rescale_X = rescale.transform(X)

x_train, x_test, y_train, y_test = train_test_split(rescale_X,Y,test_size=0.2) # 훈련용 데이터 전체의 80% 비중, 테스트 데이터 전체의비중 20%

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(y_train)
y_train = le.transform(y_train)
y_test = le.transform(y_test)

model = KNeighborsClassifier()
model.fit(x_train,y_train)
print(model.score(x_train,y_train),model.score(x_test,y_test))

from sklearn.linear_model import LogisticRegression

model1 = LogisticRegression()
model1.fit(x_train,y_train)
print(model1.score(x_train,y_train),model1.score(x_test,y_test))

from sklearn.ensemble import GradientBoostingClassifier
model2 = GradientBoostingClassifier()
model2.fit(x_train,y_train)
print(model2.score(x_train,y_train),model2.score(x_test,y_test))

from xgboost import XGBClassifier

model3 = XGBClassifier(n_estimators = 400,learning_rate = 0.0001,max_depth=7)
model3.fit(x_train,y_train)
model3.score(x_test,y_test)

from lightgbm import LGBMClassifier
model4 = LGBMClassifier(n_estimators=200)
model4.fit(x_train,y_train)
print(model4.score(x_test,y_test))
print(model4.predict(x_test),y_test)

unique, counts = np.unique(model3.predict(x_test), return_counts=True)
print(dict(zip(unique,counts)))
unique, counts = np.unique(y_test, return_counts=True)
print(dict(zip(unique,counts)))

le.classes_

unique,counts = np.unique(Y, return_counts=True)
print(dict(zip(unique,counts)))

print('Training Completed')


test = pd.read_csv('DATA 2800.csv').to_numpy()

resize_test = rescale.transform(X)
print('KNN')
print(le.inverse_transform(model.predict(resize_test)))
print('logostic RG')
print(le.inverse_transform(model1.predict(resize_test)))
print('Gradient boosting')
print(le.inverse_transform(model2.predict(resize_test)))
print('XGBoosting')
print(le.inverse_transform(model3.predict(resize_test)))
print('LightGBM')
print(le.inverse_transform(model4.predict(resize_test)))