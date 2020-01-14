import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

#load the data
oecd_bli = pd.read_csv("oecd_bli.csv", thousands=',')
gdp_per_capita = pd.read_csv("gdp_per_capita.csv", thousands=',', delimiter='\t', encoding='latin1',na_values="n/a")
country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
X = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life Satisfaction"]]

country_stats.plot(kind='scatter', x='GDP per capita', y='Life satisfaction')
plt.show()

lin_reg_model = sklearn.linear_model.LinearRegression()

lin_reg_model.fit(X,y)

X_new = [[22587]]

print(lin_reg_model.predict(X_new))
