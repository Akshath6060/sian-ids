# ==============================
# Explainable IDS Implementation
# Based on:
# Explainable AI for IDS using LIME and SHAP
# ==============================

import numpy as np
import pandas as pd
import time
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers


# ============================================
# 1️⃣ LOAD DATASET (ADFA-LD STYLE FORMAT)
# ============================================

# Dataset format assumed:
# sequence,label
# "168 45 23 23 89 12 ...",1

df = pd.read_csv("adfa_generated.csv")

X_raw = df["sequence"]
y = df["label"].values


# ============================================
# 2️⃣ PREPROCESSING (TF-IDF + 2-GRAM)
# ============================================

print("\n[INFO] Performing TF-IDF Vectorization...")

vectorizer = TfidfVectorizer(
    token_pattern=r"(?u)\b\w+\b",
    ngram_range=(2, 3), 
    max_features=200 
)

X_tfidf = vectorizer.fit_transform(X_raw).toarray()


# ============================================
# 3️⃣ FEATURE SELECTION (Chi-Square)
# ============================================

print("[INFO] Performing Chi-Square Feature Selection...")

selector = SelectKBest(chi2, k=180) 
X_selected = selector.fit_transform(X_tfidf, y)

feature_names = vectorizer.get_feature_names_out()
feature_names = feature_names[selector.get_support()]


# ============================================
# 4️⃣ TRAIN / TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42
)


# ============================================
# 5️⃣ BUILD MLP IDS MODEL
# ============================================

print("[INFO] Training MLP IDS Model...")

model = Sequential()
# Input layer
model.add(Dense(256, activation='relu', input_shape=(180,), kernel_regularizer=regularizers.l2(0.0001)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

# Hidden layers
model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(96, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
model.add(Dropout(0.15))

# Output layer
model.add(Dense(2, activation='softmax'))

model.compile(
    optimizer=Adam(learning_rate=0.0005), 
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train with history tracking
print("[INFO] Training MLP IDS Model...")
history = model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1, 
          validation_split=0.2, callbacks=[early_stop])

# Save history for visualization
import json
history_dict = {
    'loss': [float(x) for x in history.history['loss']],
    'accuracy': [float(x) for x in history.history['accuracy']],
    'val_loss': [float(x) for x in history.history['val_loss']],
    'val_accuracy': [float(x) for x in history.history['val_accuracy']]
}
with open('training_history.json', 'w') as f:
    json.dump(history_dict, f)

print("\n[INFO] Evaluating Model...")
loss, acc = model.evaluate(X_test, y_test)
print("Accuracy:", acc)

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ============================================
# 6️⃣ LIME EXPLANATION
# ============================================

print("\n[INFO] Running LIME Explanation...")

# Create prediction function that returns probabilities
def predict_fn(X):
    return model.predict(X, verbose=0)

lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train,
    feature_names=feature_names,
    class_names=["Normal", "Attack"],
    mode='classification'
)

instance_index = 0

exp = lime_explainer.explain_instance(
    X_test[instance_index],
    predict_fn,
    num_features=10
)

lime_features = exp.as_list()

print("\nTop 10 LIME Features:")
for f in lime_features:
    print(f)


# ============================================
# 7️⃣ SHAP EXPLANATION
# ============================================

print("\n[INFO] Running SHAP Explanation...")

start_time = time.time()

shap_explainer = shap.KernelExplainer(model.predict, X_train[:100])
shap_values = shap_explainer.shap_values(X_test[:10])

print("SHAP Computation Time:", time.time() - start_time)

print("\nTop SHAP Features for First Instance:")

instance_shap = shap_values[0][0]
top_indices = np.argsort(np.abs(instance_shap))[-10:]

for idx in top_indices:
    print(feature_names[idx], instance_shap[idx])


# ============================================
# 8️⃣ PERTURBATION VALIDATION (REMOVE FEATURES)
# ============================================

print("\n[INFO] Performing Perturbation Analysis...")

def perturb_instance(instance, important_feature_names):
    perturbed = instance.copy()
    for name, _ in important_feature_names:
        if name in vectorizer.vocabulary_:
            idx = vectorizer.vocabulary_[name]
            if idx < len(perturbed):
                perturbed[idx] = 0
    return perturbed


original_pred = predict_fn(X_test[instance_index].reshape(1, -1))
perturbed_instance = perturb_instance(X_test[instance_index], lime_features)
new_pred = predict_fn(perturbed_instance.reshape(1, -1))

print("\nOriginal Prediction:", np.argmax(original_pred[0]))
print("After Perturbation:", np.argmax(new_pred[0]))


# ============================================
# 9️⃣ GLOBAL PERTURBATION TEST (10 INSTANCES)
# ============================================

changed = 0

for i in range(10):
    exp = lime_explainer.explain_instance(
        X_test[i],
        predict_fn,
        num_features=10
    )
    important = exp.as_list()

    perturbed = perturb_instance(X_test[i], important)

    original = np.argmax(predict_fn(X_test[i].reshape(1,-1)), axis=1)
    new = np.argmax(predict_fn(perturbed.reshape(1,-1)), axis=1)

    if original != new:
        changed += 1

print("\nPrediction Changed in", changed, "out of 10 instances.")


# ============================================
# 🔟 LIME vs SHAP COMPUTATIONAL COMPARISON
# ============================================

print("\n[INFO] Comparing LIME vs SHAP Speed...")

start = time.time()
_ = lime_explainer.explain_instance(
    X_test[0],
    predict_fn,
    num_features=10
)
print("LIME Time:", time.time() - start)

start = time.time()
_ = shap_explainer.shap_values(X_test[:1])
print("SHAP Time:", time.time() - start)


print("\n========== IMPLEMENTATION COMPLETE ===========")