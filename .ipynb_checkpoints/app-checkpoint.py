from flask import Flask, render_template, request
import pandas as pd
import joblib
import pickle

app = Flask(__name__)

# Load the model and other necessary encoders/scalers
# model = joblib.load('model.pkl') 
# district_encoder = joblib.load('district_encoder.pkl')  
# market_encoder = joblib.load('market_encoder.pkl')    
# scaler = joblib.load('scaler.pkl')
f1 = open('model.pkl','rb')
f2 = open('district_encoder.pkl','rb')
f3 = open('market_encoder.pkl','rb')
f4 = open('scaler.pkl','rb')

model = pickle.load(f1)
district_encoder = pickle.load(f2)
market_encoder = pickle.load(f3)
scaler = pickle.load(f4)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    district = request.form['district']
    market = request.form['market']
    price_date = request.form['price_date']

    # Convert price_date to month and year
    price_date_dt = pd.to_datetime(price_date)
    month = price_date_dt.month
    year = price_date_dt.year

    # Encode district and market names
    district_encoded = district_encoder.transform([district])[0]
    market_encoded = market_encoder.transform([market])[0]

    # Prepare input DataFrame (ensure the correct order of columns)
    # Updated order to match the trained model's expected input
    input_data = pd.DataFrame([[district_encoded, market_encoded, year, month]],
                              columns=['District Name', 'Market Name', 'Year', 'Month'])  # Ensure correct order

    # Predict
    prediction = model.predict(input_data)
    
    # Inverse scale the prediction
    prediction = scaler.inverse_transform(prediction.reshape(-1, 1))[0][0]
    
    # Render the result
    return render_template('index.html', prediction=round(prediction, 2))

if __name__ == "__main__":
    app.run(debug=True)
