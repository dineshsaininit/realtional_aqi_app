from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import requests
from datetime import datetime, timedelta
import traceback
import os
import google.generativeai as genai

app = Flask(__name__)

MODEL_PATH = 'xgb_model_prediction.pkl'
try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("XGBoost Model loaded successfully.")
    else:
        model = None
        print("Warning: xgb_model_prediction.pkl not found. Using fallback math for demonstration.")
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

def get_lat_lon(city):
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
    res = requests.get(url).json()
    if "results" in res and len(res["results"]) > 0:
        data = res["results"][0]
        return data["latitude"], data["longitude"], data["name"], data.get("country", "")
    return None, None, None, None

def fetch_data_week(lat, lon):
    weather_url = 'https://api.open-meteo.com/v1/forecast'
    aqi_url = 'https://air-quality-api.open-meteo.com/v1/air-quality'

    weather_params = {
        'latitude': lat,
        'longitude': lon,
        'hourly': 'temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m',
        'timezone': 'auto',
        'forecast_days': 7
    }

    aqi_params = {
        'latitude': lat,
        'longitude': lon,
        'hourly': 'pm10,pm2_5,carbon_monoxide,nitrogen_dioxide',
        'timezone': 'auto',
        'forecast_days': 7
    }

    weather_res = requests.get(weather_url, params=weather_params).json()
    aqi_res = requests.get(aqi_url, params=aqi_params).json()

    df_weather = pd.DataFrame(weather_res["hourly"])
    df_aqi = pd.DataFrame(aqi_res["hourly"])

    df_weather['time'] = pd.to_datetime(df_weather['time'])
    df_aqi['time'] = pd.to_datetime(df_aqi['time'])

    df = pd.merge(df_aqi, df_weather, on='time')
    df.rename(columns={
        'pm2_5': 'PM2_5',
        'pm10': 'PM10',
        'carbon_monoxide': 'CO',
        'nitrogen_dioxide': 'NO2',
        'temperature_2m': 'temperature',
        'relative_humidity_2m': 'relative_humidity',
        'wind_speed_10m': 'wind_speed_num'
    }, inplace=True)

    df['hours'] = df['time'].dt.hour
    df['month'] = df['time'].dt.month
    df['day_of_week'] = df['time'].dt.day_of_week

    current = datetime.now()
    df_now_hour = df[df['hours'] == current.hour].copy()

    feature_columns = [
        'PM2_5', 'PM10', 'CO', 'NO2', 
        'temperature', 'precipitation', 'relative_humidity', 'wind_speed_num', 
        'hours', 'month', 'day_of_week'
    ]

    x_test = df_now_hour[feature_columns]

    if model is not None:
        predicted_aqi = model.predict(x_test)
    else:
        predicted_aqi = (df_now_hour['PM2_5'] * 1.8 + df_now_hour['PM10'] * 0.8).values

    final_aqi = []
    for i, aqi in enumerate(predicted_aqi):
        forecast_date = current.date() + timedelta(days=i)
        final_aqi.append({
            "date": forecast_date.strftime("%Y-%m-%d"),
            'hour': current.hour,
            'aqi': float(aqi)
        })
    
    raw_data = df_now_hour.fillna(0).to_dict(orient='records')
    for row in raw_data:
        row['time'] = row['time'].strftime("%Y-%m-%d %H:%M:%S")

    return {
        "predictions": final_aqi,
        "raw_data": raw_data
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['GET'])
def predict():
    city = request.args.get('city', 'Delhi')
    lat, lon, resolved_name, country = get_lat_lon(city)
    
    if lat is None:
        return jsonify({"error": f"City '{city}' not found. Please try another."}), 404
        
    try:
        data = fetch_data_week(lat, lon)
        return jsonify({
            "city": resolved_name,
            "country": country,
            "lat": lat,
            "lon": lon,
            "data": data
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Failed to process data: " + str(e)}), 500

@app.route('/api/ai_advice', methods=['POST'])
def ai_advice():
    data = request.json
    disease = data.get('disease')
    aqi = data.get('aqi')
    
    if not disease or not aqi:
        return jsonify({"error": "Missing disease or AQI"}), 400
        
    api_key = "AIzaSyBsLmFNtyvJ4t1PyAsN9uEDX-f72kF82_A"
    
    prompt = f"Provide the preventions, symptoms, and consequences of {disease} when the Air Quality Index (AQI) is {aqi}. Please be short, concise, and use plain formatting without heavy markdown."
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    
    try:
        genai.configure(api_key=api_key)
        
        # Try gemini-2.5-flash as a primary choice
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        response = model.generate_content(prompt)
        
        if response and response.text:
            return jsonify({"advice": response.text})
            
    except Exception as e:
        print(f"Gemini API Error with gemini-2.5-flash: {e}")
        # Fallback to gemini-3-flash
        try:
            model = genai.GenerativeModel('gemini-3-flash')
            response = model.generate_content(prompt)
            if response and response.text:
                return jsonify({"advice": response.text})
        except Exception as e2:
            print(f"Gemini API Error with gemini-3-flash: {e2}")

    return jsonify({"error": "AI Service Unavailable. Please try again later."}), 503

if __name__ == '__main__':
    app.run(debug=True, port=5000)