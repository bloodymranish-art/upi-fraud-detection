from flask import Flask, request, render_template, jsonify, redirect, url_for, session
import pickle
import pandas as pd
from tensorflow.keras.models import load_model
from preprocess import prepare_input

app = Flask(__name__)
app.secret_key = "supersecretkey"  # change in production

# ================= LOAD MODELS =================
preprocessor = pickle.load(open("models/preprocessor.pkl", "rb"))
meta = pickle.load(open("models/meta.pkl", "rb"))

rf = pickle.load(open("models/rf.pkl", "rb"))
lr = pickle.load(open("models/lr.pkl", "rb"))
knn = pickle.load(open("models/knn.pkl", "rb"))
dt = pickle.load(open("models/dt.pkl", "rb"))
gb = pickle.load(open("models/gb.pkl", "rb"))
cnn = load_model("models/cnn.h5")

FEATURES = meta["feature_columns"]

# ================= AUTH =================
USERS = {
    "admin": "admin123",
    "sonalmam":"1234"
}

# history
HISTORY = []

def is_logged_in():
    return "user" in session

def safe_float(x):
    try:
        if x is None or x == "":
            return 0.0
        return float(x)
    except Exception:
        return 0.0

def require_login():
    return is_logged_in()

# Routes

@app.route("/")
def home():
    if not is_logged_in():
        return redirect(url_for("login"))
    return render_template("index.html")

@app.route("/history")
def history():
    if not is_logged_in():
        return redirect(url_for("login"))
    return render_template("history.html")

@app.route("/settings")
def settings():
    if not is_logged_in():
        return redirect(url_for("login"))
    return render_template("settings.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        if username in USERS and USERS[username] == password:
            session["user"] = username
            return redirect(url_for("home"))
        else:
            return render_template("login.html", error="Invalid credentials")

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/history-data")
def history_data():
    if not is_logged_in():
        return jsonify({"error": "Unauthorized"}), 401
    return jsonify(HISTORY)



@app.route("/predict", methods=["POST"])
def predict():
    if not is_logged_in():
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json(silent=True) or {}
    row = data.get("features", data)

    cleaned = {}
    for k in FEATURES:
        cleaned[k] = safe_float(row.get(k, 0))

    df = pd.DataFrame([cleaned])
    df = prepare_input(df, meta)
    X = preprocessor.transform(df)

    # Model predictions
    p1 = rf.predict_proba(X)[0][1]
    p2 = lr.predict_proba(X)[0][1]
    p3 = knn.predict_proba(X)[0][1]
    p4 = dt.predict_proba(X)[0][1]
    p5 = gb.predict_proba(X)[0][1]

    X_cnn = X.reshape(-1, 2, 5, 1)
    p6 = float(cnn.predict(X_cnn, verbose=0)[0][0])

    # Ensemble
    final_score = (
        0.20 * p1 +
        0.18 * p2 +
        0.14 * p3 +
        0.14 * p4 +
        0.16 * p5 +
        0.18 * p6
    )

    prediction = "Fraud" if final_score > 0.5 else "Real"

    result = {
        "prediction": prediction,
        "confidence": round(float(final_score), 4),
        "model_scores": {
            "random_forest": round(float(p1), 4),
            "logistic_regression": round(float(p2), 4),
            "knn": round(float(p3), 4),
            "decision_tree": round(float(p4), 4),
            "gradient_boosting": round(float(p5), 4),
            "cnn": round(float(p6), 4),
        }
    }

    
    HISTORY.insert(0, {
        "id": f"#TXN-{len(HISTORY) + 1:06d}",
        "time": pd.Timestamp.now().strftime("%b %d, %Y • %H:%M:%S"),
        "user": "Admin Node #204",
        "device": "Web Dashboard",
        "amount": cleaned.get("amount", 0),
        "prediction": prediction,
        "confidence": float(final_score),
        "flags": "Auto-generated from latest analysis"
    })

    HISTORY[:] = HISTORY[:50]  

    return jsonify(result)

# 
if __name__ == "__main__":
    app.run(debug=True)