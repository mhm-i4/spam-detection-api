# 📧 Spam Detection API (FastAPI + ML from Scratch)

A machine learning-powered backend API that classifies emails as **spam or not spam**.

Built completely from scratch using:

* TF-IDF vectorization
* Logistic Regression (manual implementation)
* FastAPI for serving predictions

---

## 🚀 Features

* 📩 Classify emails as spam / not spam
* ⚡ FastAPI backend with REST endpoint
* 🧠 ML model built from scratch (no sklearn)
* 📊 Confidence score for predictions
* 🔌 Easy to extend and deploy

---

## 🧱 Tech Stack

* Python
* FastAPI
* NumPy
* Pickle (model storage)

---

## 📁 Project Structure

```
spam-detection-api/
│
├── app.py              # FastAPI backend
├── train_model.py      # Model training script
├── spam_model.pkl      # Saved model (optional)
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Run

### 1. Clone the repo

```
git clone https://github.com/your-username/spam-detection-api.git
cd spam-detection-api
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Train the model

```
python train_model.py
```

### 4. Run the API

```
uvicorn app:app --reload
```

---

## 🧪 API Usage

### Endpoint

```
POST /predict
```

### Request Body

```json
{
  "email": "win free iphone now"
}
```

### Response

```json
{
  "spam": true,
  "confidence": 0.92
}
```

---

## 🧠 How It Works

1. Email text is tokenized and cleaned
2. Converted into TF-IDF feature vector
3. Logistic regression model predicts probability
4. Output returned via API

---

## 🔥 Future Improvements

* Batch prediction support
* Better dataset (real-world emails)
* Model optimization
* Frontend integration
* Deployment (Render/AWS)

---

## 💡 Author

Built as a backend + AI/ML learning project.
