from flask import Flask, request, render_template
import pickle
import pandas as pd
from catboost import CatBoostRegressor
import json

cb_comp = CatBoostRegressor()
cb_turbine = CatBoostRegressor()

#Loading the model for both coefficients
compressor_model = cb_comp.load_model("CatBoostRegressor_GT_Compressor_model")
turbine_model = cb_turbine.load_model("CatBoostRegressor_GT_Turbine_model")

#Scaler
scaler = pickle.load(open("scaler.pickle", "rb"))



app = Flask(__name__)

@app.route("/")
def form():
    return render_template("home.html")



cols = ['Lever position (lp) [ ]', 
        'Ship speed (v) [knots]',
        'Gas Turbine shaft torque (GTT) [kN m]',
        'Gas Turbine rate of revolutions (GTn) [rpm]',
        'Gas Generator rate of revolutions (GGn) [rpm]',
        'Starboard Propeller Torque (Ts) [kN]',
        'Port Propeller Torque (Tp) [kN]',
        'HP Turbine exit temperature (T48) [C]',
        'GT Compressor inlet air temperature (T1) [C]',
        'GT Compressor outlet air temperature (T2) [C]',
        'HP Turbine exit pressure (P48) [bar]',
        'GT Compressor inlet air pressure (P1) [bar]',
        'GT Compressor outlet air pressure (P2) [bar]',
        'Gas Turbine exhaust gas pressure (Pexh) [bar]',
        'Turbine Injecton Control (TIC) [%]', 
        'Fuel flow (mf) [kg/s]']

data = pd.DataFrame(columns=cols)

data = data.append({
    'Lever position (lp) [ ]': request.form['lever'],
    'Ship speed (v) [knots]': request.form['speed'],
    'Gas Turbine shaft torque (GTT) [kN m]': request.form['turbine_shaft_torque'],
    'Gas Turbine rate of revolutions (GTn) [rpm]': request.form['turbine_rate'],
    'Gas Generator rate of revolutions (GGn) [rpm]': request.form['generator rate'],
    'Starboard Propeller Torque (Ts) [kN]': request.form['starboard_propeller'],
    'Port Propeller Torque (Tp) [kN]': request.form['port_propeller'], 
    'HP Turbine exit temperature (T48) [C]': request.form['turbine_temp'],
    'GT Compressor inlet air temperature (T1) [C]': request.form['compressor_inlet_temp'],
    'GT Compressor outlet air temperature (T2) [C]': request.form['compressor_outlet_temp'],
    'HP Turbine exit pressure (P48) [bar]': request.form['turbine_exit_pressure'],
    'GT Compressor inlet air pressure (P1) [bar]': request.form['compressor_inlet_pressure'],
    'GT Compressor outlet air pressure (P2) [bar]': request.form['compressor_outlet_pressure'],
    'Gas Turbine exhaust gas pressure (Pexh) [bar]': request.form['turbine_exhaust_pressure'],
    'Turbine Injecton Control (TIC) [%]' : request.form['turbine_injection_control'],
    'Fuel flow (mf) [kg/s]' : request.form['fuel_flow'],
}, ignore_index = True)


data_scaled = scaler.transform(data)

compressor_coefficient = compressor_model.predict(data_scaled)
turbine_coefficient = turbine_model.predict(data_scaled)

coefficients = {
    "compressor_coefficient" : compressor_coefficient ,
    "turbine_coefficient" : turbine_coefficient
}


output_values = json.dumps(coefficients)


if __name__ == "__main__":
    app.run(debug=True)