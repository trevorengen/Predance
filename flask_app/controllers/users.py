from flask import render_template, redirect, request, session
from flask_app import app
from flask_app.models.user import User
from flask_bcrypt import Bcrypt
from model_data import hdata
from model_data import mmodel
from keras.models import load_model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/load_data/<coin>/<start_range>-<end_range>')
def load_data(coin, start_range, end_range):
    model = load_model('../model_data/saved_models/' + coin)
    