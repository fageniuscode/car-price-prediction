from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Charger le modèle de prédiction des prix des voitures
model = None
with open('PredictCarPriceData/vehicle_price_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Récupérer les valeurs des attributs depuis le formulaire
    present_price = float(request.form.get('present_price'))
    kms_driven = int(request.form.get('kms_driven'))
    owner = int(request.form.get('owner'))
    years_old = int(request.form.get('years_old'))
    fuel_type_diesel = int(request.form.get('fuel_type_diesel'))
    fuel_type_petrol = int(request.form.get('fuel_type_petrol'))
    seller_type_individual = int(request.form.get('seller_type_individual'))
    transmission_manual = int(request.form.get('transmission_manual'))

    # Créer une observation avec les caractéristiques fournies
    single_obs = [[present_price, kms_driven, owner, years_old, fuel_type_diesel, fuel_type_petrol, seller_type_individual, transmission_manual]]
    df_single_obs = pd.DataFrame(single_obs, columns=['Present_Price', 'Kms_Driven', 'Owner', 'Years Old', 'Fuel_Type_Diesel', 'Fuel_Type_Petrol', 'Seller_Type_Individual', 'Transmission_Manual'])

    # Effectuer la prédiction en utilisant le modèle chargé
    predicted_price = model.predict(df_single_obs)

    # Afficher le résultat de la prédiction
    return render_template('index.html', prediction=predicted_price)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
