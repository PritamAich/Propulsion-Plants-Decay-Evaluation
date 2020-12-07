import pandas as pd
from catboost import CatBoostRegressor

import pickle

import warnings
warnings.filterwarnings("ignore")

#Enter the path of csv file here
raw_data = pd.read_csv('propulsion.csv')

data = raw_data.copy()
data = data.drop('Unnamed: 0', axis = 1)

#Splitting into independent and depenedent variables 
x = data.drop(["GT Compressor decay state coefficient.", "GT Turbine decay state coefficient."], axis = 1)
y_compressor = data['GT Compressor decay state coefficient.']
y_turbine = data['GT Turbine decay state coefficient.']

print("Independent and dependent variables created.")
print("-----------------------------------------------------------------------------------------------------------")


#Scaling the data
from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler()

x_scaled = scaler.fit_transform(x)
print("Scaling of data successfully done.")
print("-----------------------------------------------------------------------------------------------------------")


#For GT Compressor Coefficient
cb_compressor = CatBoostRegressor(bagging_temperature = 3, depth = 8, l2_leaf_reg = 0.5, learning_rate = 0.1)

print("GT Compressor Model Training in progress..........")
cb_compressor.fit(x_scaled, y_compressor, early_stopping_rounds = 10, verbose=200, plot = False) #Training the model


print("GT Compressor Model successfully trained!")

model_auth = input("Do you want to save the model? (Y/N)")

if model_auth == "Y" or model_auth == "y":

    cb_compressor.save_model("CatBoostRegressor_GT_Compressor_model")
    print("Compressor Model saved!")

print("-----------------------------------------------------------------------------------------------------------")

#For GT Compressor Turbine
cb_turbine = CatBoostRegressor(bagging_temperature = 3, depth = 8, l2_leaf_reg = 0.5, learning_rate = 0.3)

print("GT Turbine Model Training in progress..........")
cb_turbine.fit(x_scaled, y_turbine, early_stopping_rounds = 10, verbose=200, plot = False) #Training the model


print("GT Turbine Model successfully trained!")

model_auth = input("Do you want to save the model? (Y/N)")

if model_auth == "Y" or model_auth == "y":

    cb_turbine.save_model("CatBoostRegressor_GT_Turbine_model")
    print("Turbine Model saved!")


scaler_auth = input("Do you want to save the scaler object?(Y/N)")

if scaler_auth == "Y" or scaler_auth == "y":
    #Saving Scaler object
    filehandler = open('scaler.pickle', 'wb') 
    pickle.dump(scaler, filehandler)
    filehandler.close()
    print("Scaler object saved.")