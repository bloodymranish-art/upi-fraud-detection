import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from preprocess import build_dataset, build_preprocessor

os.makedirs("models", exist_ok=True)

print("Loading dataset...")
X, y, meta = build_dataset("fraud_dataset.csv")

print("Building preprocessor...")
preprocessor = build_preprocessor(meta)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train_p = preprocessor.fit_transform(X_train)
X_test_p = preprocessor.transform(X_test)

pickle.dump(preprocessor, open("models/preprocessor.pkl", "wb"))
pickle.dump(meta, open("models/meta.pkl", "wb"))

models = {
    "rf": RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),
    "lr": LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    ),
    "knn": KNeighborsClassifier(
        n_neighbors=7,
        weights="distance"
    ),
    "dt": DecisionTreeClassifier(
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42
    ),
    "gb": GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.08,
        max_depth=3,
        random_state=42
    )
}

print("\nTraining models...\n")
results = {}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_p, y_train)

    y_pred = model.predict(X_test_p)
    y_prob = model.predict_proba(X_test_p)[:, 1] if hasattr(model, "predict_proba") else y_pred

    results[name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc": roc_auc_score(y_test, y_prob)
    }

    pickle.dump(model, open(f"models/{name}.pkl", "wb"))

print("\nModel Results:")
for model_name, metrics in results.items():
    print(model_name, metrics)

print("\nTraining CNN...")


X_train_cnn = X_train_p.reshape(-1, 2, 5, 1)
X_test_cnn = X_test_p.reshape(-1, 2, 5, 1)

cnn = Sequential([
    Conv2D(32, (2, 2), activation="relu", input_shape=(2, 5, 1)),
    Dropout(0.25),
    Flatten(),
    Dense(64, activation="relu"),
    Dropout(0.25),
    Dense(1, activation="sigmoid")
])

cnn.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)

cnn.fit(
    X_train_cnn, y_train,
    validation_split=0.2,
    epochs=4,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

cnn.save("models/cnn.h5")

print("\nSaved files in models folder:")
print(os.listdir("models"))

print("\n ALL MODELS TRAINED SUCCESSFULLY!")