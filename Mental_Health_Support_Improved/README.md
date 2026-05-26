# MindEase — AI Mental Health Support Chatbot
**AI/ML Portfolio Project | Django + LSTM + NLP**

## 🧠 Tech Stack
- **Backend**: Django 4.x, Python 3.10+
- **ML Model**: TensorFlow/Keras — Stacked LSTM with LayerNormalization
- **NLP**: Tokenization, Sequence Padding, LabelEncoder (scikit-learn)
- **Frontend**: Custom HTML/CSS (dark theme, responsive)
- **Auth**: Django built-in auth with CSRF protection
- **Static Files**: WhiteNoise (production-ready)

## 🚀 Local Setup
```bash
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```
Open http://127.0.0.1:8000

## ☁️ Deploy to Render (Free)
1. Push this folder to a GitHub repo
2. Go to https://render.com → New → Web Service
3. Connect your repo
4. Set:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn Mental_Health_Support.wsgi:application`
5. Add environment variable:
   - `DJANGO_SECRET_KEY` = (generate a new secret key)
   - `DEBUG` = `False`
   - `ALLOWED_HOSTS` = `your-app.onrender.com`
6. Deploy → Done!

## ☁️ Deploy to Railway
1. Push to GitHub
2. Go to https://railway.app → New Project → GitHub Repo
3. Railway auto-detects Python/Django
4. Set same env vars as above
5. Deploy!

## 📁 Project Structure
```
Mental_Health_Support/
├── application/
│   └── views.py          # ML model loading + Django views
├── Mental_Health_Support/
│   ├── settings.py       # Production-ready settings
│   └── urls.py
├── static/
│   ├── Dataset/intents.json   # NLP training data
│   └── model/model.h5         # Pre-trained LSTM model
├── template/
│   ├── base.html         # Shared layout + navbar
│   ├── Home.html         # Landing page
│   ├── chatbot.html      # Chat UI
│   ├── login.html        # Auth
│   └── register.html     # Auth
├── requirements.txt
└── README.md
```

## 🔑 Key ML Concepts Demonstrated
- LSTM sequence modeling for NLP
- Intent classification (multi-class)
- Keras Tokenizer + pad_sequences pipeline
- Model persistence (save/load .h5)
- Inference pipeline with preprocessing consistency
- Django REST endpoint for ML model serving
