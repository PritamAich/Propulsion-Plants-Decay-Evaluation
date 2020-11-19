#This is the end file of the ML model for the GT Turbine State Decay coefficient 

#This model is considered to be performing best among six other ML models tested on this dataset.

#To recreate the model, just specify the csv file path below and run this file in the terminal and the code will do the rest.


import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error
import pickle

import warnings
warnings.filterwarnings("ignore")

#Enter the path of csv file here
raw_data = pd.read_csv('propulsion.csv')

data = raw_data.copy()
data = data.drop('Unnamed: 0', axis = 1)


#Splitting training, validation and test datasets
from sklearn.model_selection import train_test_split

x = data.drop(["GT Compressor decay state coefficient.", "GT Turbine decay state coefficient."], axis = 1)
y = data['GT Turbine decay state coefficient.']
#Training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 42)

#Training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.1, random_state = 42)

print("Training, Validation and test sets created.")
print("Model will be trained on Training set and evaluated on validation set.")
print("After evaluation on validation set, an RMSE score will be displayed for the validation set.")

print("-----------------------------------------------------------------------------------------------------------")

#Scaling the data
from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler() 
  
x_train_scaled = scaler.fit_transform(x_train)
x_val_scaled = scaler.transform(x_val)
x_test_scaled = scaler.transform(x_test) 


#CatBoost Model creation
from catboost import CatBoostRegressor

cb = CatBoostRegressor(bagging_temperature = 3, depth = 8, l2_leaf_reg = 0.5, learning_rate = 0.3) 

cb.fit(x_train_scaled, y_train, early_stopping_rounds = 10, verbose=200, plot = False) #Training the model

y_val_pred = cb.predict(x_val_scaled) #Predicting on validation set

rmse = np.sqrt(mean_squared_error(y_val, y_val_pred)) #Evaluating performance on validation set

print("RMSE score for Validation set: ", rmse)

print("-----------------------------------------------------------------------------------------------------------")


print("The model will now be tested on Test set and and RMSE score will be displayed for test set.")


#Testing model on test set
y_test_pred = cb.predict(x_test_scaled) #Testing on test set

rmse = np.sqrt(mean_squared_error(y_test, y_test_pred)) #Evaluating model rmse score by comparing actual and predicted results

print("RMSE score for Test set: ", rmse)

scaler_auth = input("Do you want to save the scaler object?(Y/N)")

if scaler_auth == "Y" or scaler_auth == "y":
    #Saving Scaler object
    filehandler = open('scaler_GT_Turbine.pickle', 'wb') 
    pickle.dump(scaler, filehandler)
    filehandler.close()
    print("Scaler object saved.")

model_auth = input("Do you want to save the model? (Y/N)")

if model_auth == "Y" or model_auth == 'y':
    #Saving the CatBoost model object
    filehandler = open('CatBoostRegressor_GT_Turbine.pickle', 'wb') 
    pickle.dump(cb, filehandler)
    filehandler.close()
    print("Model object saved.")