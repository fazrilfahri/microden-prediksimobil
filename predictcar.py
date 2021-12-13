import pandas as pd
import numpy as np 
import pickle

df=pd.read_csv('data/prediksimobil.csv')

x = df['horsepower'].values.reshape(-1,1)
y = df['price'].values.reshape(-1,1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

from sklearn.linear_model import LinearRegression
regresilinear = LinearRegression()
regresilinear.fit(x_train, y_train)

pickle.dump(regresilinear, open('predict.pkl', 'wb'))