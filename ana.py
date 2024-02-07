import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mpl_dates
import matplotlib.pyplot as plt
import datetime
from sklearn.svm import SVR


data = pd.read_csv("/Users/macuser/Desktop/buildspace/ETHUSD.csv")

#printing 10 random rows as sample
print(data.sample(10), end = '\n\n')

# printing the data description and dataset information
print(data.describe(), end = '\n\n')
print(data.info(), end = '\n\n')

#checking if there are any null values in the dataset
print(data.isnull().sum(), end = '\n\n')

#count no.of unique elements in the columns
print(data.nunique(), end = '\n\n')
  
ohlc = data.loc[:, ['Date', 'Open', 'High', 'Low', 'Close']]
ohlc['Date'] = pd.to_datetime(ohlc['Date'])
ohlc['Date'] = ohlc['Date'].apply(mpl_dates.date2num)

ohlc['Open'] = [float(str(i).replace(",", "")) for i in ohlc['Open']]
ohlc['High'] = [float(str(i).replace(",", "")) for i in ohlc['High']]
ohlc['Low'] =  [float(str(i).replace(",", "")) for i in ohlc['Low']]
ohlc['Close'] = [float(str(i).replace(",", "")) for i in ohlc['Close']]

# Creating Subplots
fig, ax = plt.subplots()
  
candlestick_ohlc(ax, ohlc.values, width=0.6, colorup='green',
                 colordown='red', alpha=0.4)
  
# Setting labels & titles
ax.set_xlabel('Date')
ax.set_ylabel('Price')
fig.suptitle('Stock Prices for a year (ETH)')
  
# Formatting Date
date_format = mpl_dates.DateFormatter('%d-%m-%Y')
ax.xaxis.set_major_formatter(date_format)
fig.autofmt_xdate()

fig.tight_layout()
  
plt.show()

ohlc['Date'] = ohlc['Date'].apply(mpl_dates.num2date)
ohlc = ohlc.set_index('Date')
# print(ohlc.head())

X = ohlc.drop('Close', axis = 1)
y = ohlc['Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = False)

#Support Vector Machine
svm = SVR(kernel = 'rbf', C = 1e3, gamma = 0.00001)
svm.fit(X_train, y_train)
print('Test accuracy of support vector machine: {}\n'.format(svm.score(X_test, y_test)*100))

pred = svm.predict(X_test)
pred = pd.Series(pred, dtype = 'float64', index = list(pd.to_datetime(y_test.index)))
# print(pred)

plt.plot(pred, label = "Prediction", lw = 2, alpha = 0.7)
plt.plot(y_test, label = 'Actual', lw = 2, alpha = 0.7)
plt.title('Prediction vs Actual (ETH)')
plt.xlabel('Date')
plt.ylabel('Price in USD')
plt.legend()
plt.xticks(rotation = 45)
plt.show()

print("\n", svm.predict((np.array([1650, 1790, 1500]).reshape(1,-1))), "\n")
