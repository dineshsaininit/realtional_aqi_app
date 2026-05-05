from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import requests
from datetime import datetime, timedelta
import traceback
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

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

WEATHER_API_KEY = os.getenv('WEATHER_API_KEY')

def get_lat_lon(city):
    # Try WeatherAPI search first
    url = f"https://api.weatherapi.com/v1/search.json?key={WEATHER_API_KEY}&q={city}"
    try:
        res = requests.get(url).json()
        if isinstance(res, list) and len(res) > 0:
            data = res[0]
            return data["lat"], data["lon"], data["name"], data.get("country", "")
    except:
        pass

    # Fallback to Open-Meteo search if WeatherAPI fails (it's better for small towns)
    try:
        url_om = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
        res_om = requests.get(url_om).json()
        if "results" in res_om and len(res_om["results"]) > 0:
            data = res_om["results"][0]
            return data["latitude"], data["longitude"], data["name"], data.get("country", "")
    except:
        pass

    return None, None, None, None

def fetch_data_week(lat, lon):
    url = f"https://api.weatherapi.com/v1/forecast.json?key={WEATHER_API_KEY}&q={lat},{lon}&days=7&aqi=yes"
    res = requests.get(url).json()

    if "error" in res:
        raise Exception(f"WeatherAPI Error: {res['error'].get('message', 'Unknown error')}")

    forecast_days = res.get("forecast", {}).get("forecastday", [])
    all_hours = []

    for day in forecast_days:
        for hour in day.get("hour", []):
            aqi_data = hour.get("air_quality", {})
            all_hours.append({
                "time": hour["time"],
                "PM2_5": aqi_data.get("pm2_5", 0),
                "PM10": aqi_data.get("pm10", 0),
                "CO": aqi_data.get("co", 0),
                "NO2": aqi_data.get("no2", 0),
                "temperature": hour["temp_c"],
                "precipitation": hour["precip_mm"],
                "relative_humidity": hour["humidity"],
                "wind_speed_num": hour["wind_kph"]
            })

    df = pd.DataFrame(all_hours)
    df['time'] = pd.to_datetime(df['time'])

    df['hours'] = df['time'].dt.hour
    df['month'] = df['time'].dt.month
    df['day_of_week'] = df['time'].dt.day_of_week

    current = datetime.now()
    # Find the data closest to current hour
    df_now_hour = df[df['hours'] == current.hour].copy()
    
    # If we have multiple days, the first one is today
    if not df_now_hour.empty:
        df_now_hour = df_now_hour.head(7) # Get 7 days of that hour for "week" forecast

    feature_columns = [
        'PM2_5', 'PM10', 'CO', 'NO2', 
        'temperature', 'precipitation', 'relative_humidity', 'wind_speed_num', 
        'hours', 'month', 'day_of_week'
    ]

    x_test = df_now_hour[feature_columns]

    if model is not None:
        try:
            predicted_aqi = model.predict(x_test)
        except Exception as e:
            print(f"Prediction error: {e}")
            predicted_aqi = (df_now_hour['PM2_5'] * 1.8 + df_now_hour['PM10'] * 0.8).values
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
        
    api_key = os.getenv('GEMINI_API_KEY')
    prompt = f"Provide the preventions, symptoms, and consequences of {disease} when the Air Quality Index (AQI) is {aqi}. Please be short, concise, and use plain formatting without heavy markdown."
    
    try:
        if not api_key:
            return jsonify({"error": "AI API Key missing. Please set GEMINI_API_KEY environment variable."}), 500
            
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