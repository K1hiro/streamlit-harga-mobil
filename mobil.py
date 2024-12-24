import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('toyota.csv')

df.head()

df.info()

sns.heatmap(df.isnull())

df.describe()

numerical_df = df.select_dtypes(include=np.number) 
sns.heatmap(numerical_df.corr(), annot=True)

models = df.groupby('model').count()[['tax']].sort_values(by='tax', ascending=True).reset_index()
models = models.rename(columns={'tax':'numberOfCars'})

fig = plt.figure(figsize=(15,5))
sns.barplot(x=models['model'], y=models['numberOfCars'], color='royalBlue')
plt.xticks(rotation=60)

engine = df.groupby('engineSize').count()[['tax']].sort_values(by='tax' ).reset_index()
engine = engine.rename(columns={'tax':'count'})

plt.figure(figsize=(15,5))
sns.barplot(x=engine['engineSize'], y=engine['count'], color='royalBlue')
plt.xticks(rotation=60)

plt.figure(figsize=(15,5))
sns.distplot(df['mileage'])

plt.figure(figsize=(15,5))
sns.distplot(df['price'])

features = ['year', 'mileage', 'tax', 'mpg', 'engineSize']
x = df[features]
y = df['price']
x.shape, y.shape

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,random_state=70)
y_test.shape

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)
pred = lr.predict(x_test)

score = lr.score(x_test, y_test)
print('akurasi model regresi linier = ',score)

input_data = np.array([[2019,5000,145,30.2,2]])
prediction = lr.predict(input_data)
print('prediksi harga mobil dalam EUR = ',prediction)

import pickle

filename = 'estimasi_mobil.sav'
pickle.dump(lr, open(filename, 'wb'))