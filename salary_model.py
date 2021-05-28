import numpy
import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv('Salary_Data.csv')

y = data['Salary']
X = data['YearsExperience']
X = X.values.reshape(-1,1)

model = LinearRegression()

model.fit(X,y)

p = int(input("Enter the experience to which salary will get predicted: "))
out = model.predict([[p]])

print(out)
