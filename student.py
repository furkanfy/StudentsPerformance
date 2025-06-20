import pandas as pd 
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

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

X = df.drop(['math score', 'passed'], axis=1)
y = df['passed']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Doğruluk Oranı:", accuracy)

df['read_write_avg'] = (df['reading score'] + df['writing score']) / 2
df['total_score'] = df['math score'] + df['reading score'] + df['writing score']
df['overall_pass'] = (df['total_score'] > 180).astype(int)

xgb = XGBClassifier(eval_metric='logloss')
xgb.fit(X_train, y_train)

y_pred = xgb.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

importance = xgb.feature_importances_
features = X_train.columns

# Görselleştirme
plt.figure(figsize=(10,6))
sns.barplot(x=importance, y=features)
plt.title("Özellik Önem Düzeyi (Feature Importance)")
plt.xlabel("Önem")
plt.ylabel("Özellikler")
plt.tight_layout()
plt.show()

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', XGBClassifier(eval_metric='logloss'))
])

param_grid = {
    'model__n_estimators': [50, 100, 200],
    'model__max_depth': [3, 5, 7],
    'model__learning_rate': [0.01, 0.1, 0.2]
}

grid = GridSearchCV(pipeline, param_grid, scoring='f1', cv=3)
grid.fit(X_train, y_train)

print("En iyi parametreler:", grid.best_params_)
print("En iyi skor:", grid.best_score_)
import joblib

# Eğitilmiş GridSearchCV pipeline'ı kaydet
joblib.dump(grid.best_estimator_, 'xgb_pipeline.pkl')