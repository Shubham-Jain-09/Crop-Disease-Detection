


# _________________------------------------------OLD CRop IMage UPLOad Code______________------------------

 
# from __future__ import division, print_function
# import sys
# import os
# import glob
# import re
# import numpy as np

# # Keras
# from keras.models import load_model
# from keras.preprocessing import image

# # Flask utils
# from flask import Flask, redirect, url_for, request, render_template
# from werkzeug.utils import secure_filename
# from gevent.pywsgi import WSGIServer

# # Define a flask app
# app = Flask(__name__)

# MODEL_PATH = 'Model.hdf5'

# # Load your trained model
# print(" ** Model Loading **")
# model = load_model(MODEL_PATH)
# print(" ** Model Loaded **")
# model._make_predict_function()



# def model_predict(img_path, model):
#     img = image.load_img(img_path, target_size=(224, 224))

#     # Preprocessing the image
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)

#     x = x/255

#     preds = model.predict(x)
#     d = preds.flatten()
#     j = d.max()
#     li=['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
#     for index,item in enumerate(d):
#         if item == j:
#             class_name = li[index].split('___')
#     return class_name


# @app.route('/')
# def index():
#     # Main page
#     return render_template('index.html')


# @app.route('/predict', methods=['GET', 'POST'])
# def upload():
#     if request.method == 'POST':
#         # Get the file from post request
#         f = request.files['file']

#         # Save the file to ./uploads
#         basepath = os.path.dirname(__file__)
#         file_path = os.path.join(
#             basepath, 'uploads', secure_filename(f.filename))
#         f.save(file_path)

#         # Make prediction
#         class_name = model_predict(file_path, model)

#         result = str(f"Predicted Crop:{class_name[0]}  Predicted Disease:{class_name[1].title().replace('_',' ')}")               
#         return result
#     return None


# if __name__ == '__main__':
#     app.run(debug=True)


# ---------------------------------------New Code  CROP Disease Prediction Code________----------------------------------------------------------------------


from __future__ import division, print_function
import io
import numpy as np
from flask import Flask, request, render_template, jsonify, redirect, url_for ,session , flash
from werkzeug.utils import secure_filename
from transformers import pipeline
import torch
import requests
from PIL import Image
import base64
from torchvision import transforms
from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic
from markupsafe import Markup
import config
from utils.model import ResNet9
import pickle
import redis
import json
import firebase_admin
from firebase_admin import credentials
import time
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, auth
from firebase_admin import credentials, auth, firestore
from flask import Flask, request,render_template, redirect,session
from flask_sqlalchemy import SQLAlchemy
import bcrypt
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.models import load_model
# from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import load_img, img_to_array
import pandas as pd
from dotenv import load_dotenv
import h5py
import tempfile
# ---------------------old Pipelines for prediction--------------------------------------

# pipe = pipeline("image-classification", model="SanketJadhav/PlantDiseaseClassifier-Resnet50")
# pipe = pipeline("image-classification", model="ayerr/plant-disease-classification")

# ---------------------old Pipelines for prediction--------------------------------------


# -----------------------------------------new Pipeline for accurate prediction--------------------
pipe = pipeline("object-detection", model="krifa/LeafDiseaseDetection")
# -----------------------------------------new Pipeline for accurate prediction--------------------

#  Firebase auth initialization


# cred = credentials.Certificate("reboot.json")
# # firebase_admin.initialize_app(cred)


# #  firestore client  initilaziation

# db = firestore.client()


# # Loading disease Classes

disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

# disease_model_path = 'models/plant_disease_model.pth'
# disease_model = ResNet9(3, len(disease_classes))
# disease_model.load_state_dict(torch.load(
#     disease_model_path, map_location=torch.device('cpu')))
# disease_model.eval()


# Loading crop recommendation model

crop_recommendation_model_path = 'D:\\new_crop\\Apna_kisan_MVp\\Apna_kisan_MVP\\model\\RandomForest.pkl'
crop_recommendation_model = pickle.load(
    open(crop_recommendation_model_path, 'rb'))


soil_type_prediction_model_path = 'D:\\new_crop\\Apna_kisan_MVp\\Apna_kisan_MVP\\model\\DenseNet121v2_95.h5'

labels = ['Chalky Soil', 'Mary Soil', 'Sand', 'Slit Soil', 'Alluvial Soil', 'Black Soil', 'Clay Soil', 'Red Soil']

# Load models with custom_objects to handle layer name issues
def load_model_safely(model_path):
    """Load model with error handling for layer name issues"""
    try:
        # Try loading with compile=False first
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        print(f"Standard loading failed: {e}")
        try:
            # Try with custom_objects to handle layer name issues
            model = tf.keras.models.load_model(model_path, compile=False, custom_objects={})
            return model
        except Exception as e2:
            print(f"Custom objects loading failed: {e2}")
            # Create a fallback model
            print("Creating fallback model...")
            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(4, activation='softmax')  # 4 soil types
            ])
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            return model

soil_model = load_model_safely(soil_type_prediction_model_path)

# Load the model once and reuse it
model_path = "D:\\new_crop\\Apna_kisan_MVp\\Apna_kisan_MVP\\model\\SoilNet_93_86.h5"
SoilNet = load_model_safely(model_path)

# Soil types and corresponding crop recommendations
classes = {
    0: "Alluvial Soil:-{ Rice, Wheat, Sugarcane, Maize, Cotton, Soyabean, Jute }",
    1: "Black Soil:-{ Virginia, Wheat, Jowar, Millets, Linseed, Castor, Sunflower }",
    2: "Clay Soil:-{ Rice, Lettuce, Chard, Broccoli, Cabbage, Snap Beans }",
    3: "Red Soil:-{ Cotton, Wheat, Pulses, Millets, Oil Seeds, Potatoes }"
}


# app = Flask(__name__)
# API key for OpenWeatherMap
# weather_api_key =''

load_dotenv()  # Load environment variables from .env

weather_api_key = os.getenv("OPEN_WEATHER_APIKEY")
print(weather_api_key)

def get_weather_data(latitude, longitude, api_key=weather_api_key):
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return {
            'temperature': data['main']['temp'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'wind_speed': data['wind']['speed']
        }
    else:
        return None
    


recommendations = {
    'Apple___Apple_scab': {
        'fertilizer': 'Copper-based Fungicide',
        'details': 'Apply 150ml per plant if high humidity.',
        'application_method': 'Mix with water and spray.'
    },
    'Apple___Black_rot': {
        'fertilizer': 'Copper-based Fungicide',
        'details': 'Apply 150ml per plant if high humidity.',
        'application_method': 'Mix with water and spray.'
    },
    'Apple___Cedar_apple_rust': {
        'fertilizer': 'General-purpose Fertilizer',
        'details': 'Apply 50g per plant, every two weeks.',
        'application_method': 'Spread around base.'
    },
    'Blueberry___healthy': {
        'fertilizer': 'General-purpose Fertilizer',
        'details': 'Apply 50g per plant, every two weeks.',
        'application_method': 'Spread around base.'
    },
    'Cherry_(including_sour)___Powdery_mildew': {
        'fertilizer': 'Sulfur-based Fungicide',
        'details': 'Use 100g per plant if high temperature.',
        'application_method': 'Dilute in 1L of water and spray thoroughly.'
    },
    'Cherry_(including_sour)___healthy': {
        'fertilizer': 'General-purpose Fertilizer',
        'details': 'Apply 50g per plant, every two weeks.',
        'application_method': 'Spread around base.'
    },
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {
        'fertilizer': 'General-purpose Fertilizer',
        'details': 'Apply 50g per plant, every two weeks.',
        'application_method': 'Spread around base.'
    },
    'Corn_(maize)___Common_rust_': {
        'fertilizer': 'General-purpose Fertilizer',
        'details': 'Apply 50g per plant, every two weeks.',
        'application_method': 'Spread around base.'
    },
    'Corn_(maize)___Northern_Leaf_Blight': {
        'fertilizer': 'General-purpose Fertilizer',
        'details': 'Apply 50g per plant, every two weeks.',
        'application_method': 'Spread around base.'
    },
    'Corn_(maize)___healthy': {
        'fertilizer': 'General-purpose Fertilizer',
        'details': 'Apply 50g per plant, every two weeks.',
        'application_method': 'Spread around base.'
    },
    'Grape___Black_rot': {
        'fertilizer': 'Copper-based Fungicide',
        'details': 'Apply 150ml per plant if high humidity.',
        'application_method': 'Mix with water and spray.'
    },
    'Grape___Esca_(Black_Measles)': {
        'fertilizer': 'General-purpose Fertilizer',
        'details': 'Apply 50g per plant, every two weeks.',
        'application_method': 'Spread around base.'
    },
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {
        'fertilizer': 'General-purpose Fertilizer',
        'details': 'Apply 50g per plant, every two weeks.',
        'application_method': 'Spread around base.'
    },
    'Grape___healthy': {
        'fertilizer': 'General-purpose Fertilizer',
        'details': 'Apply 50g per plant, every two weeks.',
        'application_method': 'Spread around base.'
    },
    'Orange___Haunglongbing_(Citrus_greening)': {
        'fertilizer': 'General-purpose Fertilizer',
        'details': 'Apply 50g per plant, every two weeks.',
        'application_method': 'Spread around base.'
    },
    'Peach___Bacterial_spot': {
        'fertilizer': 'Copper-based Fungicide',
        'details': 'Apply 150ml per plant if high humidity.',
        'application_method': 'Mix with water and spray.'
    },
    'Peach___healthy': {
        'fertilizer': 'General-purpose Fertilizer',
        'details': 'Apply 50g per plant, every two weeks.',
        'application_method': 'Spread around base.'
    },
    'Pepper,_bell___Bacterial_spot': {
        'fertilizer': 'Copper-based Fungicide',
        'details': 'Apply 150ml per plant if high humidity.',
        'application_method': 'Mix with water and spray.'
    },
    'Pepper,_bell___healthy': {
        'fertilizer': 'General-purpose Fertilizer',
        'details': 'Apply 50g per plant, every two weeks.',
        'application_method': 'Spread around base.'
    },
    'Potato___Early_blight': {
        'fertilizer': 'Copper-based Fungicide',
        'details': 'Apply 150ml per plant if high humidity.',
        'application_method': 'Mix with water and spray.'
    },
    'Potato___Late_blight': {
        'fertilizer': 'Copper-based Fungicide',
        'details': 'Apply 150ml per plant if high humidity.',
        'application_method': 'Mix with water and spray.'
    },
    'Potato___healthy': {
        'fertilizer': 'General-purpose Fertilizer',
        'details': 'Apply 50g per plant, every two weeks.',
        'application_method': 'Spread around base.'
    },
    'Raspberry___healthy': {
        'fertilizer': 'General-purpose Fertilizer',
        'details': 'Apply 50g per plant, every two weeks.',
        'application_method': 'Spread around base.'
    },
    'Soybean___healthy': {
        'fertilizer': 'General-purpose Fertilizer',
        'details': 'Apply 50g per plant, every two weeks.',
        'application_method': 'Spread around base.'
    },
    'Squash___Powdery_mildew': {
        'fertilizer': 'Sulfur-based Fungicide',
        'details': 'Use 100g per plant if high temperature.',
        'application_method': 'Dilute in 1L of water and spray thoroughly.'
    },
    'Strawberry___Leaf_scorch': {
        'fertilizer': 'General-purpose Fertilizer',
        'details': 'Apply 50g per plant, every two weeks.',
        'application_method': 'Spread around base.'
    },
    'Strawberry___healthy': {
        'fertilizer': 'General-purpose Fertilizer',
        'details': 'Apply 50g per plant, every two weeks.',
        'application_method': 'Spread around base.'
    },
    'Tomato___Bacterial_spot': {
        'fertilizer': 'Copper-based Fungicide',
        'details': 'Apply 150ml per plant if high humidity.',
        'application_method': 'Mix with water and spray.'
    },
    'Tomato___Early_blight': {
        'fertilizer': 'Copper-based Fungicide',
        'details': 'Apply 150ml per plant if high humidity.',
        'application_method': 'Mix with water and spray.'
    },
    'Tomato___Late_blight': {
        'fertilizer': 'Copper-based Fungicide',
        'details': 'Apply 150ml per plant if high humidity.',
        'application_method': 'Mix with water and spray.'
    },
    'Tomato___Leaf_Mold': {
        'fertilizer': 'Copper-based Fungicide',
        'details': 'Apply 150ml per plant if high humidity.',
        'application_method': 'Mix with water and spray.'
    },
    'Tomato___Septoria_leaf_spot': {
        'fertilizer': 'Copper-based Fungicide',
        'details': 'Apply 150ml per plant if high humidity.',
        'application_method': 'Mix with water and spray.'
    },
    'Tomato___Spider_mites Two-spotted_spider_mite': {
        'fertilizer': 'General-purpose Fertilizer',
        'details': 'Apply 50g per plant, every two weeks.',
        'application_method': 'Spread around base.'
    },
    'Tomato___Target_Spot': {
        'fertilizer': 'Copper-based Fungicide',
        'details': 'Apply 150ml per plant if high humidity.',
        'application_method': 'Mix with water and spray.'
    },
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        'fertilizer': 'General-purpose Fertilizer',
        'details': 'Apply 50g per plant, every two weeks.',
        'application_method': 'Spread around base.'
    },
    'Tomato___Tomato_mosaic_virus': {
        'fertilizer': 'General-purpose Fertilizer',
        'details': 'Apply 50g per plant, every two weeks.',
        'application_method': 'Spread around base.'
    },
    'Tomato___healthy': {
        'fertilizer': 'General-purpose Fertilizer',
        'details': 'Apply 50g per plant, every two weeks.',
        'application_method': 'Spread around base.'
    }
}



def generate_fertilizer_recommendation(disease_name, latitude, longitude, api_key):
    weather_data = get_weather_data(latitude, longitude, api_key)
    
    if not weather_data:
        return {
            'fertilizer': 'General-purpose Fertilizer',
            'details': 'Could not fetch weather data. Use general-purpose fertilizer as per standard guidelines.',
            'application_method': 'Spread evenly around the base of the plant and water thoroughly.'
        }
    
    temperature = weather_data['temperature']
    humidity = weather_data['humidity']

    if "blight" in disease_name.lower() and humidity > 80:
        return {
            'fertilizer': 'Copper-based Fungicide',
            'details': f'Apply 150ml per plant. The high humidity ({humidity}%) suggests an increased risk, so apply every 7 days.',
            'application_method': 'Mix with water and spray evenly over the leaves.'
        }
    elif "mildew" in disease_name.lower() and temperature > 30:
        return {
            'fertilizer': 'Sulfur-based Fungicide',
            'details': f'Use 100g per plant. The high temperature ({temperature}°C) requires reapplication every 5 days.',
            'application_method': 'Dilute in 1L of water and spray thoroughly.'
        }
    else:
        return {
            'fertilizer': 'General-purpose Fertilizer',
            'details': 'Apply 50g per plant, once every two weeks during the growing season.',
            'application_method': 'Spread evenly around the base of the plant and water thoroughly.'
        }

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}




def predict_image(img, model=pipe):
    """
    Transforms image to tensor and predicts disease label
    :params: image
    :return: prediction (string)
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    yb = model(img_u)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    # Retrieve the class label
    return prediction


def preprocess_image(image_path):
    """
    Preprocess the image to match the model input requirements.
    """
    img = Image.open(image_path)
    img = img.resize((224, 224))  # Resize the image to the model's input size
    img = np.array(img) / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def model_predict(image_path, model):
    """
    Predicts the soil type and suggests crops based on the model prediction.
    """
    try:
        image = load_img(image_path, target_size=(224, 224))
        image = img_to_array(image)
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        
        result = np.argmax(model.predict(image))
        prediction = classes[result]
        
        if result == 0:
            return "Alluvial", "Alluvial.html"
        elif result == 1:
            return "Black", "Black.html"
        elif result == 2:
            return "Clay", "Clay.html"
        elif result == 3:
            return "Red", "Red.html"
    except Exception as e:
        print(f"Error: {e}")  # Log the error for debugging
        return "Sorry we couldn't process your request currently. Please try again", None


def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city.
    :params: city_name
    :return: temperature, humidity or None if there's an issue
    """
    api_key = config.weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()

    if x["cod"] != "404":
        y = x.get("main", {})

        # Safely get the temperature and convert from Kelvin to Celsius
        temp_kelvin = y.get("temp")
        if temp_kelvin is not None:
            try:
                temperature = round(temp_kelvin - 273.15, 2)
            except TypeError as e:
                print(f"Error rounding temperature: {e}")
                temperature = None
        else:
            print(f"Temperature data not found in API response for city: {city_name}")
            temperature = None

        # Safely get the humidity
        humidity = y.get("humidity")
        if humidity is None:
            print(f"Humidity data not found in API response for city: {city_name}")

        # Return only if both temperature and humidity are valid
        if temperature is not None and humidity is not None:
            return temperature, humidity
        else:
            return None
    else:
        print(f"City {city_name} not found or API error occurred")
        return None


# Set the allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load the image classification pipeline
pipe = pipeline("image-classification", model="SanketJadhav/PlantDiseaseClassifier-Resnet50")

# Check if the uploaded file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



# Define a Flask app
app = Flask(__name__ , static_folder='static')



# app.secret_key = 'apna_kisan'  # Use a strong secret key
CORS(app)  # Enable CORS if needed
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///farmers_database.db'
app.config['UPLOAD_FOLDER'] = 'uploads/'
db = SQLAlchemy(app)
app.secret_key = 'apna_kisan'




class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))

    def __init__(self,email,password,name):
        self.name = name
        self.email = email
        self.password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def check_password(self,password):
        return bcrypt.checkpw(password.encode('utf-8'),self.password.encode('utf-8'))
    


with app.app_context():
    db.create_all()

# Define routes for login, signup, and logout


# @app.route('/')
# def index1():
#     return render_template('index1.html')

@app.route('/signup', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']


 
        # Check if the user already exists
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash('Email address already registered.', 'error')
            return render_template('signup.html')

        # Create new user
        new_user = User(name=name, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()

        flash('Registration successful! Please log in.', 'success')
        return redirect('/login')

    return render_template('signup.html')




@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = User.query.filter_by(email=email).first()

        if user and user.check_password(password):
            session['email'] = user.email
            flash(f"User {user.email} logged in successfully.", 'success')
            return redirect('/dashboard')
        else:
            flash('Invalid email or password.', 'error')
            return render_template('login.html')

    return render_template('login.html')


@app.route('/dashboard')
def dashboard():
    if 'email' in session:
        print(f"Session email: {session['email']}")  # Debugging
        user = User.query.filter_by(email=session['email']).first()
        if user:
            print(f"Rendering dashboard for {user.email}.")  # Debugging
            return render_template('dashboard.html', user=user)
    
    print("Redirecting to login.")  # Debugging
    flash("You need to log in first.", 'warning')
    return redirect('/login')


@app.route('/logout')
def logout():
    session.pop('email', None)
    flash('You have been logged out.', 'info')
    return redirect('/login')


@app.route('/')
def index():
    # Main page (Home)
    return render_template('index.html')


@app.route('/home')
def home():
    # Redirect to index for the Home link
    return redirect(url_for('index'))

@ app.route('/crop-recommend')
def crop_recommend():
    title = 'Arogya Krishi - Crop Recommendation'
    return render_template('crop.html', title=title)


@app.route('/soil-predict', methods=['POST'])
def soil_prediction():
    title = 'Arogya Krishi - Soil Prediction'

    if 'soil_image' not in request.files:
        return render_template('try_again.html', title=title, error_message="No file part in the request.")

    file = request.files['soil_image']
    if file.filename == '':
        return render_template('try_again.html', title=title, error_message="No file selected for uploading.")

    if file:
        try:
            # Secure and save the file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print(f"File path: {file_path}")  # Debugging line to check file path
            file.save(file_path)

            # Use the model_predict function to predict soil type and recommend crop
            predicted_soil, template_name = model_predict(file_path, SoilNet)
            print(f"Predicted Soil: {predicted_soil}, Template: {template_name}")  # Debugging line

            return render_template(template_name, prediction=predicted_soil, title=title)

        except Exception as e:
            print(f"Error: {str(e)}")  # Print the error to the console for debugging
            return render_template('try_again.html', title=title, error_message=f"An error occurred: {str(e)}")

    return render_template('try_again.html', title=title, error_message="Something went wrong.")



@ app.route('/fertilizer')
def fertilizer_recommendation():
    title = 'Arogya Krishi - Fertilizer Suggestion'

    return render_template('fertilizer.html', title=title)

@ app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    title = 'Arogya Krishi - Fertilizer Suggestion'

    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])
    # ph = float(request.form['ph'])

    df = pd.read_csv('D:\\new_crop\\Apna_kisan_MVp\\Apna_kisan_MVP\data\\fertilizer.csv')

    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = nr - N
    p = pr - P
    k = kr - K
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]
    if max_value == "N":
        if n < 0:
            key = 'NHigh'
        else:
            key = "Nlow"
    elif max_value == "P":
        if p < 0:
            key = 'PHigh'
        else:
            key = "Plow"
    else:
        if k < 0:
            key = 'KHigh'
        else:
            key = "Klow"

    response = Markup(str(fertilizer_dic[key]))

    return render_template('fertilizer-result.html', recommendation=response, title=title)

# render disease prediction result page


# render disease prediction result page

# @app.route('/disease')
# def disease_prediction():
#     # Disease prediction page
#     return render_template('disease.html')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400
        
        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No file selected for uploading'}), 400

        if file and allowed_file(file.filename):
            # Load the image
            img = Image.open(file).convert('RGB')

            # Convert the image to a base64 string
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

            # Use the pipeline to make a prediction
            prediction = pipe(img)
            print(prediction)  # Debugging line
            
            # Get the highest probability label
            if prediction:
                top_prediction = max(prediction, key=lambda x: x['score'])
                return render_template('crop-result.html', prediction=top_prediction, image_base64=img_base64)
            else:
                return jsonify({'error': 'No predictions made'}), 500
        else:
            return jsonify({'error': 'Allowed file types are png, jpg, jpeg'}), 400
        

@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'Arogya Krishi - Disease Detection'

    if request.method == 'POST':
        if 'file' not in request.files:
            error = 'No file part in the request'
            return render_template('disease.html', title=title, error=error)
        
        file = request.files.get('file')
        if not file or not allowed_file(file.filename):
            error = 'Allowed file types are png, jpg, jpeg'
            return render_template('disease.html', title=title, error=error)

        try:
            # Load the image
            img = Image.open(file).convert('RGB')

            # Convert the image to a base64 string
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

            # Use the pipeline to make a prediction
            prediction = pipe(img)
            print("Prediction:", prediction)  # Debugging line

            # Get the highest probability label
            if prediction:
                top_prediction = max(prediction, key=lambda x: x['score'])
                disease_name = top_prediction['label']

                # Example: set fixed latitude and longitude for now
                latitude = 19.0760
                longitude = 72.8777

                # Generate the fertilizer recommendation dynamically
                fertilizer_info = generate_fertilizer_recommendation(disease_name, latitude, longitude, weather_api_key)

                return render_template('disease-result.html', 
                                       prediction=top_prediction, 
                                       fertilizer=fertilizer_info, 
                                       image_base64=img_base64, 
                                       title=title)
            else:
                error = 'No predictions made. Please try again with a different image.'
                return render_template('disease.html', title=title, error=error)
        
        except Exception as e:
            print(f"Error: {e}")  # Log the error for debugging
            error = 'An error occurred during prediction. Please try again.'
            return render_template('disease.html', title=title, error=error)

    return render_template('disease.html', title=title)


if __name__ == '__main__':
    app.run(debug=True)





