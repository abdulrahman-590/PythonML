import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
import os
from joblib import dump, load

  

def downloadData(fileName, url):
    print("Downloading Data...")
    response = requests.get(url)

    if response.status_code == 200:
        
        with open(fileName, "wb") as f:
            f.write(response.content)

        print("Data Saved")

def plotGraph(x,y, xlabel, ylabel):
    plt.scatter(x, y, color="blue")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    
def dataExploration(data):

    print("\nData Size:")
    print(data.shape)
    
    print("\nData Sample:")
    print(data.head(5))
    
    
    print("\nData Stats:")
    print(data.describe())

    df = data[["ENGINESIZE", "FUELCONSUMPTION_CITY", "FUELCONSUMPTION_HWY", "FUELCONSUMPTION_COMB", "FUELCONSUMPTION_COMB_MPG", "CO2EMISSIONS"]]

    x = df.FUELCONSUMPTION_HWY
    y = df.CO2EMISSIONS


    for column in df.columns:
        x = df[[column]]
        plotGraph(x,y, column, "Emission")



url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
fileName = "../datasets/EmissionData.csv"

# downloadData(fileName, url)

data = pd.read_csv(fileName)
df = data[["ENGINESIZE", "FUELCONSUMPTION_CITY", "FUELCONSUMPTION_HWY", "FUELCONSUMPTION_COMB", "FUELCONSUMPTION_COMB_MPG", "CO2EMISSIONS","CYLINDERS"]]
# dataExploration(data)

# Train/Test Split Alocation
split = np.random.rand(len(df)) < 0.8
train = df[split]
test = df[~split]

# Train/Test Data Setup:
# Simple Linear Regression

# x = "FUELCONSUMPTION_COMB"
# y = "CO2EMISSIONS"

# train_x = train[[x]]
# train_y = train[[y]]

# test_x = test[[x]]
# test_y = test[[y]]


# Multiple Linear Regression
x = ["FUELCONSUMPTION_COMB", "ENGINESIZE", "CYLINDERS"]
y = "CO2EMISSIONS"

# train_x = np.array(train[x])  # can also use simple list
# train_y = np.array(train[[y]])

# test_x = np.array(test[x])
# test_y = np.array(test[[y]])

# # Training
# regr = linear_model.LinearRegression()
# regr.fit(train_x, train_y)

# test_prediction = regr.predict(test_x)

# os.system("clear")
# print("\n")
# print("Mean Absolute Error:", metrics.mean_absolute_error(test_y, test_prediction))
# print("Mean Squared Error:", metrics.mean_squared_error(test_y, test_prediction))
# print("Regr Score:", regr.score(test_x, test_y)) # Explained variance score: 1 is perfect prediction


# dataToPredict = pd.read_csv("PredData.csv")
# df = dataToPredict[["ENGINESIZE", "FUELCONSUMPTION_CITY", "FUELCONSUMPTION_HWY", "FUELCONSUMPTION_COMB", "FUELCONSUMPTION_COMB_MPG", "CO2EMISSIONS","CYLINDERS"]]
# pred_x = np.array(df[x])
# actual_y = np.array(df[[y]])

# prediction = regr.predict(pred_x)

# print("\n")
# for emission in prediction:
#     print("Predictions", emission)
# print("Mean Absolute Error:", metrics.mean_absolute_error(actual_y, prediction))
# print("Mean Squared Error:", metrics.mean_squared_error(actual_y, prediction))
# print("Regr Score:", regr.score(pred_x, actual_y)) # Explained variance score: 1 is perfect prediction
# print("\n")



k = 5
batchStart = 0
limit = int(len(df) / k)

regr = linear_model.LinearRegression()
dfUse = df.copy()


for i in range(0, k):
    if i > 0:
        regr = load("model.joblib")
    batchLimit = batchStart + limit
    test = dfUse[batchStart:batchLimit]
    t1 = dfUse[:batchStart]
    t2 = dfUse[batchLimit:]
    train = pd.concat([t1, t2])

    train_x = train[x]
    train_y = train[[y]]

    test_x = test[x]
    test_y = test[[y]]
    
    regr.fit(train_x, train_y)

    predicted_y = regr.predict(test_x)

    print("\nMean Absolute Error:", metrics.mean_absolute_error(test_y, predicted_y))
    print("Mean Squared Error:", metrics.mean_squared_error(test_y, predicted_y))
    print("Regr Score:", regr.score(test_x, test_y)) # Explained variance score: 1 is perfect prediction

    batchStart = batchLimit
    saveModel = dump(regr, "model.joblib")

# predicted_y = regr.predict(pred_x)
# print("Mean Absolute Error:", metrics.mean_absolute_error(actual_y, predicted_y))
# print("Mean Squared Error:", metrics.mean_squared_error(actual_y, predicted_y))
# print("Regr Score:", regr.score(pred_x, actual_y)) # Explained variance score: 1 is perfect prediction





