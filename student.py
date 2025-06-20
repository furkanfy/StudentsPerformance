import pandas as pd 
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df=pd.read_csv('C:/Users/ASUS/Documents/student/StudentsPerformance.csv')
print(df.head())

df.info()
df.describe()
print(df.isnull().sum())
df=pd.get_dummies(df, 
                    columns=['gender', 'race/ethnicity', 'parental level of education', 'lunch',
       'test preparation course'],
                    drop_first=True,dtype=np.uint8)

df.head()
df['passed'] = (df['math score'] > 60).astype(int)
X = df.drop(['math score', 'passed'], axis=1)
y = df['passed']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
log = LogisticRegression(max_iter=1000)
model=log.fit(X_train, y_train)
model.score(X_test,y_test)