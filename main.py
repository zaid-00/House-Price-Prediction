# Importing Libraries
from sklearn.linear_model import Lasso
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# Reading data from the file
data = pd.read_csv('kc_house_data.csv')

# Analyzing data
print(data.head())
print(data.shape)
print(data.columns)
print(data.isnull().sum())
print(data.tail())
print(data.info())
print(data.describe())

# Dropping useless columns
data = data.drop('id', axis='columns')
data = data.drop('date', axis='columns')

# Dividing data to X and Y
y = data['price']
X = data.iloc[:, 1:]

# Making Standard Scaler instance
sc = StandardScaler()
kf = KFold(n_splits=3)

# Feature Scaling
X = sc.fit_transform(X)

# Making model of Random Forest
rf_model = RandomForestRegressor(n_estimators=10, random_state=42)

# Making model of Random Forest
svr_model = svm.SVR(kernel='linear', C=1)

# Making model of Random Forest
lasso_model = Lasso(alpha=0.1)

# Lists for error and score
score_for_random_forest = []
score_for_svr = []
score_for_lasso = []
error_for_random_forest = []
error_for_svr = []
error_for_lasso = []
for train_index, test_index in kf.split(X):
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]

    # Training the model
    rf_model.fit(X_train, y_train)
    # Getting the prediction for Random Forest Regressor
    predictions_for_random_forest = rf_model.predict(X_test)
    # Calculating mean_squared_error_for_random_forest
    error_for_random_forest.append(mean_squared_error(y_test, predictions_for_random_forest))
    # Calculating r2_score_for_random_forest
    score_for_random_forest.append(r2_score(y_test, predictions_for_random_forest))

    # Training the model
    svr_model.fit(X_train, y_train)
    # Getting the prediction for Support Vector Regressor
    predictions_for_svr = svr_model.predict(X_test)
    # Calculating mean_squared_error_for_svr
    error_for_svr.append(mean_squared_error(y_test, predictions_for_svr))
    # Calculating r2_score_for_svr
    score_for_svr.append(r2_score(y_test, predictions_for_svr))

    # Training the model
    lasso_model.fit(X_train, y_train)
    # Getting the prediction for Lasso
    predictions_for_lasso = lasso_model.predict(X_test)
    # Calculating mean_squared_error_for_lasso
    error_for_lasso.append(mean_squared_error(y_test, predictions_for_lasso))
    # Calculating r2_score_for_lasso
    score_for_lasso.append(r2_score(y_test, predictions_for_lasso))

# Storing values of score to plot them
names = ["r2_score_for_random_forest", "r2_score_for_svr", "r2_score_for_lasso"]
values = [np.mean(score_for_random_forest), np.mean(score_for_svr), np.mean(score_for_lasso)]
# Plotting using bar
plt.bar(names, values)
plt.show()

# Storing values of error to plot them
names = ["mean_squared_error_for_random_forest", "mean_squared_error_for_svr", "mean_squared_error_for_lasso"]
values = [np.mean(error_for_random_forest), np.mean(error_for_svr), np.mean(error_for_lasso)]
# Plotting using bar
plt.bar(names, values)
plt.show()
