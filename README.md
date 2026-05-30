# UPI Fraud Detection — Paytm Sentinel 🛡️

A production-grade **real-time UPI transaction fraud detection system** built with a weighted ensemble of 6 machine learning models and deployed as a Flask REST API with a professional fintech dashboard UI.

## 🚀 Live Demo

> Run locally using the setup instructions below.
> Login credentials: `admin / admin123`

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Total Transactions | 26,393 |
| Features | 64 input features |
| Fraud Cases | 4,545 (17.2%) |
| Real Cases | 21,848 (82.8%) |
| Missing Values | None |
| Amount Range | ₹139 — ₹49,970 |

### Feature Categories:
- **Transaction features** — amount, velocity, time of day
- **Device/IP features** — unusual device flag, IP flag, location flag
- **Behavioral biometrics** — keyboard speed, input timing, app switching frequency
- **OTP analysis** — request frequency, device consistency, timing
- **UPI handle features** — handle age, typo analysis, verification status
- **Social engineering signals** — time pressure indicators, relationship to requester

---

## 🧠 Model Architecture

### Weighted Ensemble of 6 Models

| Model | Weight | Description |
|-------|--------|-------------|
| Random Forest | 0.20 | 200 estimators, max depth 15, balanced class weight |
| Logistic Regression | 0.18 | 2000 iterations, balanced class weight |
| CNN | 0.18 | Conv2D on reshaped feature matrix (2×5×1) |
| Gradient Boosting | 0.16 | 150 estimators, learning rate 0.08 |
| KNN | 0.14 | 7 neighbors, distance-weighted |
| Decision Tree | 0.14 | Max depth 12, balanced class weight |

### Ensemble Formula:
```
final_score = 0.20×RF + 0.18×LR + 0.18×CNN + 0.16×GB + 0.14×KNN + 0.14×DT
prediction  = "Fraud" if final_score > 0.5 else "Real"
```

### CNN Architecture:
```
Input (2×5×1) → Conv2D(32, 2×2, ReLU) → Dropout(0.25)
→ Flatten → Dense(64, ReLU) → Dropout(0.25) → Dense(1, Sigmoid)
```

---

## 🏗️ System Architecture

```
fraud_dataset.csv
      │
      ▼
preprocess.py ──── build_dataset() ──── feature engineering
      │             build_preprocessor() ── StandardScaler + OneHotEncoder
      ▼
model_training.py ── train 5 sklearn models + CNN
      │               save to models/ as .pkl and .h5
      ▼
app.py (Flask REST API)
      │
      ├── POST /predict ──── weighted ensemble inference
      ├── GET  /history ──── transaction audit log
      ├── GET  /settings ─── ML threshold configuration
      └── POST /login ────── session-based authentication
```

---

## 🖥️ Dashboard Features

- **Real-time fraud prediction** with confidence scores per model
- **Transaction history** with risk badges and confidence bars
- **ML threshold settings** — configure auto-block and review triggers
- **Paytm Sentinel UI** — professional fintech dashboard design
- **Per-model breakdown** — see individual scores from all 6 models

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | Python, Flask |
| ML Models | Scikit-learn, TensorFlow/Keras |
| Data Processing | Pandas, NumPy |
| Preprocessing | StandardScaler, OneHotEncoder |
| Frontend | HTML, Tailwind CSS, Vanilla JS |
| Model Storage | Pickle (.pkl), HDF5 (.h5) |

---

## ⚙️ Setup Instructions

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/bloodymranish-art/upi-fraud-detection.git
cd upi-fraud-detection

# Install dependencies
pip install -r requirements.txt

# Train the models (first time only)
python model_training.py

# Run the Flask app
python app.py
```

Open browser at: `http://localhost:5000`

Login with: `admin / admin123`

---

## 📡 API Reference

### POST /predict

Predict whether a transaction is fraudulent.

**Request:**
```json
{
  "features": {
    "amount": 15000.00,
    "session_duration": 45,
    "authentication_attempts": 3,
    "receiver_account_age": 120,
    "geographic_disparity": 1,
    "transaction_time_of_day": 2,
    "unusual_device_flag": 1,
    "unusual_ip_flag": 0,
    "unusual_location_flag": 1,
    "transaction_velocity": 5
  }
}
```

**Response:**
```json
{
  "prediction": "Fraud",
  "confidence": 0.8732,
  "model_scores": {
    "random_forest": 0.91,
    "logistic_regression": 0.85,
    "knn": 0.79,
    "decision_tree": 0.88,
    "gradient_boosting": 0.92,
    "cnn": 0.84
  }
}
```

### GET /history-data

Returns last 50 transaction predictions.

### POST /login

```json
{ "username": "admin", "password": "admin123" }
```

---

## 📁 Project Structure

```
upi-fraud-detection/
├── app.py                 # Flask REST API + routing
├── model_training.py      # Model training pipeline
├── preprocess.py          # Data preprocessing utilities
├── fraud_dataset.csv      # 26,393 UPI transactions
├── requirements.txt       # Python dependencies
├── models/                # Trained model files (generated)
│   ├── rf.pkl
│   ├── lr.pkl
│   ├── knn.pkl
│   ├── dt.pkl
│   ├── gb.pkl
│   ├── cnn.h5
│   ├── preprocessor.pkl
│   └── meta.pkl
└── templates/
    ├── index.html         # Main dashboard
    ├── history.html       # Transaction logs
    ├── login.html         # Authentication
    └── settings.html      # ML threshold settings
```

---

## 🔑 Key Design Decisions

### Why Weighted Ensemble?
Single models suffer from bias-variance tradeoffs. The weighted ensemble combines the strengths of tree-based models (RF, GB, DT), linear models (LR), instance-based (KNN), and deep learning (CNN) to achieve more robust fraud detection.

### Why CNN on Tabular Data?
The 10 features are reshaped into a 2×5×1 matrix, allowing the CNN to learn spatial relationships between feature groups (e.g., device features vs behavioral features).

### Why class_weight="balanced"?
The dataset has 17.2% fraud rate — imbalanced classes. Balanced weighting prevents models from predicting "Real" for everything.

### Why stratify=y in train_test_split?
Ensures the 17.2% fraud ratio is preserved in both train and test sets for reliable evaluation.

---

## 📈 Model Performance

Run `python model_training.py` to see per-model metrics including accuracy, precision, recall, F1, and AUC-ROC scores printed to console.

---

## Author

Anish — B.Tech CSE, IIIT Bhopal (2023–2027)

GitHub: [@bloodymranish-art](https://github.com/bloodymranish-art)
