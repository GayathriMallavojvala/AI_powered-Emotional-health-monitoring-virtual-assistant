"""
Mental Health Support - Views
AI/ML Chatbot using LSTM + NLP with Django backend.
Author: Portfolio Project | AI/ML Engineer
"""

from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth import login, logout, authenticate
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import json
import re
import random
import os

# ──────────────────────────────────────────────
#  Load & preprocess dataset at module level
# ──────────────────────────────────────────────
DATASET_PATH = os.path.join('static', 'Dataset', 'intents.json')
MODEL_PATH   = os.path.join('static', 'model', 'model.h5')

with open(DATASET_PATH, 'r') as f:
    data = json.load(f)

df_raw = pd.DataFrame(data['intents'])
dic = {"tag": [], "patterns": [], "responses": []}
for i in range(len(df_raw)):
    ptrns = df_raw.iloc[i]['patterns']
    rspns = df_raw.iloc[i]['responses']
    tag   = df_raw.iloc[i]['tag']
    for p in ptrns:
        dic['tag'].append(tag)
        dic['patterns'].append(p)
        dic['responses'].append(rspns)

df = pd.DataFrame.from_dict(dic)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

tokenizer = Tokenizer(lower=True, split=' ')
tokenizer.fit_on_texts(df['patterns'])
vocab_size = len(tokenizer.word_index)

ptrn2seq = tokenizer.texts_to_sequences(df['patterns'])
X = pad_sequences(ptrn2seq, padding='post')

lbl_enc = LabelEncoder()
y = lbl_enc.fit_transform(df['tag'])

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Embedding, LSTM, LayerNormalization, Dense, Dropout

# Load or train model
if os.path.exists(MODEL_PATH):
    model = keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully.")
else:
    print("⚙️  Training new model...")
    model = Sequential([
        Input(shape=(X.shape[1],)),
        Embedding(input_dim=vocab_size + 1, output_dim=100, mask_zero=True),
        LSTM(32, return_sequences=True),
        LayerNormalization(),
        LSTM(32, return_sequences=True),
        LayerNormalization(),
        LSTM(32),
        LayerNormalization(),
        Dense(128, activation="relu"),
        LayerNormalization(),
        Dropout(0.2),
        Dense(128, activation="relu"),
        LayerNormalization(),
        Dropout(0.2),
        Dense(len(np.unique(y)), activation="softmax"),
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, y, batch_size=10, epochs=50,
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=3)])
    model.save(MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")


# ──────────────────────────────────────────────
#  Prediction helper
# ──────────────────────────────────────────────
def generate_answer(pattern: str) -> str:
    """Predict intent tag and return a random response."""
    txt = re.sub(r"[^a-zA-Z']", ' ', pattern).lower().split()
    txt = " ".join(txt)
    x_test = tokenizer.texts_to_sequences([txt])
    x_test = np.array(x_test).squeeze()
    x_test = pad_sequences([x_test], padding='post', maxlen=X.shape[1])
    y_pred = model.predict(x_test, verbose=0).argmax()
    tag = lbl_enc.inverse_transform([y_pred])[0]
    responses = df[df['tag'] == tag]['responses'].values[0]
    return random.choice(responses)


# ──────────────────────────────────────────────
#  Views
# ──────────────────────────────────────────────
def home(request):
    return render(request, 'Home.html')


def register(request):
    if request.method == 'POST':
        name     = request.POST.get('name', '').strip()
        email    = request.POST.get('email', '').strip()
        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '')
        cnfm     = request.POST.get('cnfm_password', '')

        if password != cnfm:
            messages.error(request, 'Passwords do not match.')
            return render(request, 'register.html')
        if User.objects.filter(username=username).exists():
            messages.error(request, 'Username already taken.')
            return render(request, 'register.html')
        if User.objects.filter(email=email).exists():
            messages.error(request, 'Email already registered.')
            return render(request, 'register.html')
        if len(password) < 8:
            messages.error(request, 'Password must be at least 8 characters.')
            return render(request, 'register.html')

        user = User.objects.create_user(
            username=username, password=password,
            email=email, first_name=name
        )
        user.save()
        messages.success(request, 'Account created! Please log in.')
        return redirect('login')

    return render(request, 'register.html')


def login_view(request):
    if request.user.is_authenticated:
        return redirect('chatbot')
    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '')
        user = authenticate(username=username, password=password)
        if user is not None:
            login(request, user)
            messages.success(request, f'Welcome back, {user.first_name or user.username}!')
            return redirect('chatbot')
        messages.error(request, 'Invalid username or password.')
    return render(request, 'login.html')


def logout_view(request):
    logout(request)
    messages.info(request, 'You have been logged out.')
    return redirect('login')


@login_required(login_url='login')
def chatbot_view(request):
    if request.method == 'POST':
        user_message = request.POST.get('message', '').strip()
        if not user_message:
            return JsonResponse({'response': 'Please type a message.'})
        bot_response = generate_answer(user_message)
        return JsonResponse({
            'response': bot_response,
            'status': 'success'
        })
    return render(request, 'chatbot.html')
