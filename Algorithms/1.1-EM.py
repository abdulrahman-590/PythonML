import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import KFold
import os
from joblib import dump, load

data = pd.read_csv("EmissionData.csv")
df = data[["ENGINESIZE", "FUELCONSUMPTION_CITY", "FUELCONSUMPTION_HWY", "FUELCONSUMPTION_COMB", "FUELCONSUMPTION_COMB_MPG", "CO2EMISSIONS","CYLINDERS"]]

kf = KFold(n_splits=6, shuffle=True, random_state=42)

x = np.array(df[["FUELCONSUMPTION_COMB", "ENGINESIZE", "CYLINDERS"]])
y = np.array(df[["CO2EMISSIONS"]])



model = linear_model.LinearRegression()
run = False

for train_index, test_index in kf.split(x):
    if run == True:
        model = load("kfmodel.joblib")
    X_train, X_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]


    model.fit(X_train, y_train)


    y_pred = model.predict(X_test)
    mse = metrics.mean_squared_error(y_test, y_pred)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    print("\n")
    print("MSE:",mse)
    print("MAE:",mae)

    saveModel = dump(model, "kfmodel.joblib")
    run = True