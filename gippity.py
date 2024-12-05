from sklearn.linear_model import LinearRegression
from sklearn import metrics
from joblib import dump
import pandas as pd
import numpy as np


data = pd.read_csv("EmissionData.csv")
df = data[["ENGINESIZE", "FUELCONSUMPTION_CITY", "FUELCONSUMPTION_HWY", "FUELCONSUMPTION_COMB", "FUELCONSUMPTION_COMB_MPG", "CO2EMISSIONS","CYLINDERS"]]


x = np.array(df[["FUELCONSUMPTION_COMB", "ENGINESIZE", "CYLINDERS"]])
y = np.array(df[["CO2EMISSIONS"]])



# K-Fold Cross-Validation function
def k_fold_cross_validation(X, y, k=5):
    """
    Performs K-Fold Cross Validation on a dataset using Linear Regression.

    Parameters:
    - X: Feature matrix (numpy array or pandas DataFrame).
    - y: Target vector (numpy array or pandas Series).
    - k: Number of folds (default=5).

    Returns:
    - avg_mse: Average Mean Squared Error across all folds.
    - fold_mse: List of MSEs for each fold.
    """
    n_samples = X.shape[0]
    indices = np.arange(n_samples)  # Create indices for splitting
    np.random.shuffle(indices)     # Shuffle data indices

    fold_size = n_samples // k
    fold_mse = []

    for i in range(k):
        # Define test fold indices
        start, end = i * fold_size, (i + 1) * fold_size
        test_indices = indices[start:end]
        train_indices = np.concatenate((indices[:start], indices[end:]))

        # Split the data
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        # Train Linear Regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict on the test set
        y_pred = model.predict(X_test)

        # Calculate Mean Squared Error
        mse = metrics.mean_squared_error(y_test, y_pred)
        fold_mse.append(mse)

    # Return average MSE and all fold MSEs
    avg_mse = np.mean(fold_mse)
    return avg_mse, fold_mse


# Generate synthetic data for demonstration


# Perform K-Fold Cross Validation
avg_mse, fold_mse = k_fold_cross_validation(x, y, k=5)

print(f"Mean Squared Error for each fold: {fold_mse}")
print(f"Average Mean Squared Error: {avg_mse}")
