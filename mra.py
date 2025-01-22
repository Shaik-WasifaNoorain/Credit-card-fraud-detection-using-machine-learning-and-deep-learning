import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import pickle


df  = pd.read_csv('C:/Users/Vaishnavi/OneDrive/Desktop/flaskmp/Maternal Health Risk Data Set.csv')
#df.info
#df.head()
#df['RiskLevel'].head()
df = df.replace({'RiskLevel':{'low risk':0, 'mid risk':1 , 'high risk':2}})
#df.head()
X = df.drop(columns=['RiskLevel'],axis=1)
y = df['RiskLevel']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1,stratify=y,random_state=2)
regressor = RandomForestRegressor(n_estimators=100)
regressor.fit(X_train,y_train)
test_data_prediction = regressor.predict(X_test)

classifier = RandomForestClassifier()
rfc=classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
accuracy = accuracy_score(predictions, y_test)
#print("Accuracy:", accuracy)

pickle.dump(rfc, open('mra.pkl', 'wb'))