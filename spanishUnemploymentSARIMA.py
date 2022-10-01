# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 12:30:40 2022

@author: pablo
"""

#Math Exam 5 Time Series

# Base -----------------------------------------------------------
import numpy as np
import pandas as pd
import statsmodels.api as sm
import os

# Viz ------------------------------------------------------------
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 5
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from plotnine import *


# Date -----------------------------------------------------------
from datetime import datetime
import calendar

# Model ----------------------------------------------------------
from statsmodels.formula.api import ols
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from scipy.signal import detrend

#Data ------------------------------------------------------------
data = pd.read_csv("Quarterly.csv", sep=",")

#%%
"Q1"
data["month"] = data.Quarter * 3
data["YearMonth"] = ""


for index in range(data.shape[0]):
    data.YearMonth[index] = str(data.Year[index]) + "-" + str(data.month[index])
    
data["YearMonth"]= pd.to_datetime(data.YearMonth, format = "%Y-%m")
data.set_index(keys = data.YearMonth, inplace = True, drop = True, append = False)

data.sort_values(by=["Year", "Quarter"], inplace = True)
#Q1

#matplotlib
plt.plot(data.YearMonth, data.qUR, linestyle = "-", label= "Unemployment Rate", color = "blue")
plt.axhline(y=data.qUR.mean(), color='b', linestyle='dotted', label = "Avg UR")
plt.plot(data.YearMonth, data.qIR, linestyle = "-", label = "Infaltion Rate", color = "red")
plt.axhline(y=data.qIR.mean(), color='r', linestyle='dotted', label = "Avg IR")
plt.title("Unemployment and Inflation Rates in Spain")
plt.legend()
plt.show()



"""looks like the Unemployment rates aimplitude may change with time, so I will transform it into a log,
to see if that changes anything. Amplitude for inflation rate stayed for or less the same, so I have
determined that it is additive."""

data["lqUR"] = np.log(data["qUR"])
plt.plot(data.YearMonth, data.lqUR, linestyle = "-", label= "Unemployment Rate", color = "blue")
plt.title("Unemployment Rate transformed into Log")
plt.legend()
plt.show()


"""Transforming it into log has not made a change in the amplitude, I must look at the seasonality
and residual components to see if the Unmployment Rate is additive or multiplicative (geometric). 
If the residuals and seasonalities are independent of the trend, then qUR is additive."""

#%%
"Q2"
#verify that it is additive
"Seasonal Decomposition for Spanish Inflation Rate"
result=seasonal_decompose(data['qIR'], model= "additive", period=12)
#plt.title("Seasonal Decomposition for Spanish Inflation Rate")     
result.plot()

"Seasonal Decomposition for Spanish Unemployment Rate"
result_UR=seasonal_decompose(data['qUR'], model= "additive", period=12)   
#plt.title("Seasonal Decomposition for Spanish unemployment Rate")      
result_UR.plot()

"""
qUR's seasonality and residuals(noise) look to be independent from the trend, so
I have determined that it is additive. The three components, trend, seasonality, and the
residuals, when added make up the data. What is first apparent in the seasonal decompositon
of the Unemployment Rate is the seasonality - the data experiences an obvious repeating pattern.
Similarly, the inflation rate also experiences seasonality, but more frequently. 

When comparing the trends, they go in accordance with basic economic theory - that unemployment 
and inflation have an inverse relationship. The seasonal decompositions lead me to believe
that a fall in unemployment may lead to a rise in inflation, or vice versa. 

Given that there the time series do experience trends this may be due to the moving averages.
I will make moving average graphs to verify if the time series do change statistically over time.
"""

#choosing a periodicity of 12 quarters, so 3 years for Inflation Rate and Unemployment Rate
m_aver_ur = data['qUR'].rolling(12).mean()
std_aver_ur = data['qUR'].rolling(12).std()
orig = plt.plot(data['qUR'], color='blue',label='Original')
mean = plt.plot(m_aver_ur, color='red', label='Moving Average')
std = plt.plot(std_aver_ur, color='black', label = 'Moving Standard Deviation')
plt.legend(loc='best')
plt.title('Moving Average & Standard Deviation - Original Unemployment Rate')
plt.show(block=False)

m_aver_ir = data['qIR'].rolling(12).mean()
std_aver_ir = data['qIR'].rolling(12).std()
orig = plt.plot(data['qIR'], color='blue',label='Original')
mean = plt.plot(m_aver_ir, color='red', label='Moving Average')
std = plt.plot(std_aver_ir, color='black', label = 'Moving Standard Deviation')
plt.legend(loc='best')
plt.title('Moving Average & Standard Deviation - Original Inflation Rate')
plt.show(block=False)

""""clearly, the unemployment time series statistical changes over time so time 
series is non-stationary and experiences a trend. However, for the inflation rate
the moving average and moving std does move around the 0 mark. I need a Dickey Fuller
test to make a stationary conclusion on the inflation.
"""


#%%
"Q3"
"""For a time series to be stationary its statistical values must not change over time. 
I am aiming for stationarity as it becomes a lot easier to predict when a time series is
stationary"""

"Dickey Fuller Test before detrending"
inflation_rates = data.qIR
result_if = adfuller(inflation_rates)
print('p-value for inflation rate: %f' % result_if[1])

unemployment_rates = data.qUR
result_ur = adfuller(unemployment_rates)
print('p-value for unemployment rate: %f' % result_ur[1])

"""Detrending the series by an order of 1 for the inflation and an order of
 two for the unemployment rate, given that it is polynomial."""


data['differenceIR'] = data['qIR'] - data['qIR'].shift()
data["differenceUR"] = data['qUR'] - data['qUR'].shift()

plt.plot(data['differenceIR'])
plt.title('DifferencedIR')

plt.plot(data['differenceUR'])
plt.title('DifferencedUR')



"The trend has been removed from both time series;however, seasonality still remains."
"Dickey Fuller Test after detrending"
inflation_rates_differenced = data.differenceIR.dropna()
result_if_differenced = adfuller(inflation_rates_differenced)
print('p-value for inflation rate: %f' % result_if_differenced[1])

unemployment_rates_differenced = data.differenceUR.dropna()
result_ur_differenced = adfuller(unemployment_rates_differenced)
print('p-value for unemployment rate: %f' % result_ur_differenced[1])




"""After detrending the inflation rate time series, we have made it stationary. However,
for the unemployment rate, the ADF test still returns that it is not stationary (cannot reject null)
for a 5%. This may be due to the seasonality seen in the differencedUR plot.Though, for a 20%
can we say that it is stationary.
"""

#%%
"Seasonal Plot"
ggplot(aes(x="Quarter", y="qUR", group = "Year", color = "Year"),data) + geom_line() +\
    geom_point(size = 2 , color= ['darkred' for value in list(data['Quarter'])]) + ggtitle("Unemployment Rate per Year in Spain") + labs(y ="Unemployment Rate")

ggplot(aes(x="Quarter", y="qIR", group = "Year", color = "Year"),data) + geom_line() +\
    geom_point(size = 2 , color= ['darkred' for value in list(data['Quarter'])]) +ggtitle("Inflation Rate per Year in Spain") + labs(y ="Inflation Rate")

#%%
"Q's 5 and 6"
plt.subplot(211)
plot_acf(data['qUR'].dropna(), lags = 32, ax = plt.gca())
plt.ylabel('ACF')

plt.subplot(212)
plot_pacf(data['qUR'].dropna(), lags = 32, ax = plt.gca())
plt.ylabel('PACF')
plt.xlabel('Lags', fontsize = 15)

plt.tight_layout(rect = (0,0,1,0.94))
plt.title("ACF and PACF curves for UR")
plt.show()

plt.subplot(211)
plot_acf(data['qIR'].dropna(), lags = 32, ax = plt.gca())
plt.ylabel('ACF')

plt.subplot(212)
plot_pacf(data['qIR'].dropna(), lags = 32, ax = plt.gca())
plt.ylabel('PACF')
plt.xlabel('Lags', fontsize = 15)

plt.tight_layout(rect = (0,0,1,0.94))
plt.title("ACF and PACF curves for IR")
plt.show()

"""For the Inflation Rate the PACF of 29 tells us that there is a stastical significant
negative impact of the unemployment rate 29 quarters ago on the inflation rate today. Similairly, 
the unemployment rate experiences the same behavior. The pacf suggests that the unemployment rate
29 quarters ago is highly correlated with the current one, more precisely, in the negative direction. 
With this knowledge can we determine a good time series model that lets us predict the unemployment and
inflation rate - we must include the IR and UR of 28 quarters ago. The pacf plot indicates that the p, in 
pdq for both time series is 29. 

The fact that ACF decreases relatively slowly may support the non-stationary status of the Unemployment
Rate time series. When looking at the moving averages,standard deviations and the Dickey Fuller test calculated
for the UR time series, they all supported the notion of it not being stationary. 

For both time series, the plots indicate that the q values, due to the ACF tailing off, are 0. 

For the original component, not the seasonal, the PACF and ACF indicate of the ARIMA model are

(29,1,0) for the Inflation Rate
(29,2,0) for the Unemployment Rate

The integrated part of the (S)ARIMA model suggests that there our time series data experiences some type
of trend, either upward or downward or both. This was calculated earlier. Let's take a look at the differenced time
 """

#PACF and ACFs curves for detrended time series
plt.subplot(211)
plot_acf(data['differenceUR'].dropna(), lags = 32, ax = plt.gca())
plt.ylabel('ACF')

plt.subplot(212)
plot_pacf(data['differenceUR'].dropna(), lags = 30, ax = plt.gca())
plt.ylabel('PACF')
plt.xlabel('Lags', fontsize = 15)

plt.tight_layout(rect = (0,0,1,0.94))
plt.title("Differenced ACF and PACF curves for UR")
plt.show()

plt.subplot(211)
plot_acf(data['differenceIR'].dropna(), lags = 30, ax = plt.gca())
plt.ylabel('ACF')

plt.subplot(212)
plot_pacf(data['differenceIR'].dropna(), lags = 30, ax = plt.gca())
plt.ylabel('PACF')
plt.xlabel('Lags', fontsize = 15)

plt.tight_layout(rect = (0,0,1,0.94))
plt.title("Differenced ACF and PACF curves for IR")
plt.show()


#PACF and ACF curves for detrended and deseasonalized
plt.subplot(211)
plot_acf(data['differenceUR'].diff(4).dropna(), lags = 32, ax = plt.gca())
plt.ylabel('ACF')

plt.subplot(212)
plot_pacf(data['differenceUR'].diff(4).dropna(), lags = 30, ax = plt.gca())
plt.ylabel('PACF')
plt.xlabel('Lags', fontsize = 15)

plt.tight_layout(rect = (0,0,1,0.94))
plt.title("Differenced and Deseaonalized ACF and PACF curves for UR")
plt.show()

plt.subplot(211)
plot_acf(data['differenceIR'].diff(4).dropna(), lags = 30, ax = plt.gca())
plt.ylabel('ACF')

plt.subplot(212)
plot_pacf(data['differenceIR'].diff(4).dropna(), lags = 30, ax = plt.gca())
plt.ylabel('PACF')
plt.xlabel('Lags', fontsize = 15)

plt.tight_layout(rect = (0,0,1,0.94))
plt.title("Differenced and Deseasonalized ACF and PACF curves for IR")
plt.show()

#%%
"Q7 and Q8"

#creating the (S)ARIMA model with the Unemployment Rate as an exogenous variable
sarima_1 = sm.tsa.SARIMAX(data['qIR'], exog = data["qUR"], order = (0,1,1), seasonal_order = (0,1,3,4)).fit()
print(sarima_1.summary())

"the Moving Average S.L12 coeffecient may be irrelevant due to the p value being greater than 0.05, so I will drop it"
sarima_2 = sm.tsa.SARIMAX(data['qIR'], exog = data["qUR"], order = (0,1,1), seasonal_order = (0,1,2,4)).fit()
print(sarima_2.summary())

"the Moving Average S.L8 coeffecient may be irrelevant due to the p value being greater than 0.05, so I will drop it"
sarima_3 = sm.tsa.SARIMAX(data['qIR'], exog = data["qUR"], order = (0,1,1), seasonal_order = (0,1,1,4)).fit()
print(sarima_3.summary())

"Removing Unemployment Rate as an exogenous variable"
sarima_4 = sm.tsa.SARIMAX(data['qIR'], order = (0,1,1), seasonal_order = (0,1,1,4)).fit()
print(sarima_4.summary())


#%%
"Q9 - Predicting using both Sarima models"

prediction = pd.DataFrame({'Inflation Rate': sarima_3.predict(n_periods=185)}, 
                          index = data['qIR'].index)
prediction_2 = pd.DataFrame({'Inflation Rate': sarima_4.predict(n_periods=185)}, 
                          index = data['qIR'].index)

plt.plot(data['qIR'], label='Original')
plt.plot(prediction["Inflation Rate"], label='SARIMA(0,1,1)(0,1,1)4UR')
plt.plot(prediction_2["Inflation Rate"], label='SARIMA(0,1,1)(0,1,1)4noUR')
plt.legend()
plt.show()

"Q10 answered on blackboard, no code needed"
