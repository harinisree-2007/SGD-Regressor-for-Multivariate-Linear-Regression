# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1: Define the Problem
- Goal: Predict two target variables:
- House Price
- Number of Occupants
- Input features (examples):
- Size of house (sq. ft.)
- Number of rooms
- Location index
- Age of house
- Amenities (garage, garden, etc.)

Step 2: Collect & Prepare Data
- Gather dataset with columns like:
| Size (sq.ft.) | Rooms | Age | Location | Price | Occupants | |---------------|-------|-----|----------|-------|-----------| | 1200          | 3     | 10  | Urban    | 50L   | 4         | | 2000          | 4     | 5   | Suburban | 80L   | 5         |
- Preprocess:
- Handle missing values
- Encode categorical variables (e.g., Location → numeric)
- Normalize/scale features for SGD efficiency
Step 3: Split Dataset
- Training set (e.g., 80%)
- Test set (e.g., 20%)

Step 4: Initialize SGD Regressor
- Choose parameters:
- loss = 'squared_loss' (for regression)
- penalty = 'l2' (regularization)
- learning_rate = 'optimal' or fixed
- max_iter = 1000

Step 5: Train the Model
- Fit the model on training data:
- Input: Features (X)
- Output: Targets (Y = [Price, Occupants])

Step 6: Prediction
- For new house data (size, rooms, age, location, etc.), predict:
- House Price
- Number of Occupants
Step 7: Evaluate Model
- Metrics:
- Mean Squared Error (MSE)
- R² Score
- Evaluate separately for Price and Occupants predictions




 

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: 
RegisterNumber:  
*/
```

import numpy as np 
from sklearn.datasets import fetch_california_housing 
from sklearn.linear_model import SGDRegressor 
from sklearn.multioutput import MultiOutputRegressor 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error 
from sklearn.preprocessing import StandardScaler 
 
# Load the California Housing dataset 
data = fetch_california_housing() 
 
# Use the first 3 features as inputs 
X = data.data[:, :3]  # Features: 'MedInc', 'HouseAge', 'AveRooms' 
 
# Use 'MedHouseVal' and 'AveOccup' as output variables 
Y = np.column_stack((data.target, data.data[:, 6]))  # Targets: 'MedHouseVal', 'AveOccup' 
                                                                                                                                                                                               
 
 
 
# Split the data into training and testing sets 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42) 
 
# Scale the features and target variables 
scaler_X = StandardScaler() 
scaler_Y = StandardScaler() 
 
X_train = scaler_X.fit_transform(X_train) 
X_test = scaler_X.transform(X_test) 
Y_train = scaler_Y.fit_transform(Y_train) 
Y_test = scaler_Y.transform(Y_test) 
 
# Initialize the SGDRegressor 
sgd = SGDRegressor(max_iter=1000, tol=1e-3) 
 
# Use MultiOutputRegressor to handle multiple output variables 
multi_output_sgd = MultiOutputRegressor(sgd) 
 
# Train the model 
multi_output_sgd.fit(X_train, Y_train) 
 
# Predict on the test data 
Y_pred = multi_output_sgd.predict(X_test) 
 
# Inverse transform the predictions to get them back to the original scale 
Y_pred = scaler_Y.inverse_transform(Y_pred) 
Y_test = scaler_Y.inverse_transform(Y_test) 
 
# Evaluate the model using Mean Squared Error 
mse = mean_squared_error(Y_test, Y_pred) 
print("Mean Squared Error:", mse) 
 
# Optionally, print some predictions 
print("\nPredictions:\n", Y_pred[:5])  # Print first 5 predictions

## Output:

multivariate linear regression model for predicting the price of the house and number of occupants in the house

![WhatsApp Image 2025-12-08 at 22 35 04_2b0d1843](https://github.com/user-attachments/assets/4ed472ca-d700-4bdf-970b-6ec13e1fff68)



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
