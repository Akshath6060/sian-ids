import numpy as np
import pandas as pd
import joblib
import os
import json
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import lime.lime_tabular
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variables to store model and preprocessors
model = None
vectorizer = None
selector = None
lime_explainer = None
feature_names = None

def load_models():
    """Load or train model if needed"""
    global model, vectorizer, selector, lime_explainer, feature_names
    
    try:
        # Check if model files exist
        if os.path.exists('model.h5') and os.path.exists('vectorizer.pkl') and os.path.exists('selector.pkl'):
            print("[INFO] Loading existing model...")
            model = load_model('model.h5')
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            vectorizer = joblib.load('vectorizer.pkl')
            selector = joblib.load('selector.pkl')
        else:
            print("[INFO] Training new model...")
            # Load dataset
            df = pd.read_csv("adfa_generated.csv")
            X_raw = df["sequence"]
            y = df["label"].values
            
            # TF-IDF Vectorization
            vectorizer = TfidfVectorizer(
                analyzer='word',
                ngram_range=(1, 3),
                max_features=300
            )
            X_tfidf = vectorizer.fit_transform(X_raw).toarray()
            
            # Feature Selection
            selector = SelectKBest(chi2, k=200)
            X_selected = selector.fit_transform(X_tfidf, y)
            
            # Train/Test Split
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y, test_size=0.2, random_state=42
            )
            
            # Build and train model
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
            from tensorflow.keras.optimizers import Adam
            from tensorflow.keras.callbacks import EarlyStopping
            from tensorflow.keras import regularizers
            
            model = Sequential()
            model.add(Dense(512, activation='relu', input_shape=(200,), kernel_regularizer=regularizers.l2(0.0001)))
            model.add(BatchNormalization())
            model.add(Dropout(0.25))
            
            model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
            model.add(BatchNormalization())
            model.add(Dropout(0.25))
            
            model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))
            
            model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
            model.add(BatchNormalization())
            model.add(Dropout(0.15))
            
            model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
            model.add(Dropout(0.1))
            
            model.add(Dense(2, activation='softmax'))
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0, 
                     validation_split=0.2, callbacks=[early_stop])
            
            # Save models
            model.save('model.h5')
            joblib.dump(vectorizer, 'vectorizer.pkl')
            joblib.dump(selector, 'selector.pkl')
        
        feature_names = vectorizer.get_feature_names_out()
        
        # Initialize LIME explainer
        df = pd.read_csv("adfa_generated.csv")
        X_raw = df["sequence"]
        X_tfidf = vectorizer.transform(X_raw).toarray()
        X_selected = selector.transform(X_tfidf)
        
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_selected[:100],
            feature_names=feature_names,
            class_names=["Normal", "Attack"],
            mode='classification'
        )
        
        print("[INFO] Model loaded successfully!")
        
    except Exception as e:
        print(f"[ERROR] Failed to load model: {str(e)}")

def predict_fn(X):
    """Prediction function for LIME"""
    return model.predict(X, verbose=0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        sequence_str = data.get('sequence', '').strip()
        
        if not sequence_str:
            return jsonify({'error': 'Please enter a syscall sequence'}), 400
        
        # Preprocess input
        try:
            X_tfidf = vectorizer.transform([sequence_str]).toarray()
            X_selected = selector.transform(X_tfidf)
        except Exception as ve:
            print(f"[ERROR] Vectorization error: {str(ve)}")
            return jsonify({'error': f'Invalid sequence format: {str(ve)}'}), 400
        
        # Make prediction
        prediction = model.predict(X_selected, verbose=0)
        pred_class = int(np.argmax(prediction[0]))
        confidence = float(np.max(prediction[0]))
        
        class_names = {0: "Normal Traffic", 1: "Attack Traffic"}
        pred_label = class_names[pred_class]
        
        # Get LIME explanation
        try:
            lime_exp = lime_explainer.explain_instance(
                X_selected[0],
                predict_fn,
                num_features=10,
                top_labels=None
            )
            lime_features = lime_exp.as_list()
        except Exception as lime_err:
            print(f"[WARNING] LIME error: {str(lime_err)}")
            lime_features = [("Feature analysis unavailable", 0.0)]
        
        return jsonify({
            'prediction': pred_label,
            'confidence': round(confidence * 100, 2),
            'class': pred_class,
            'lime_features': lime_features
        })
    
    except Exception as e:
        print(f"[ERROR] Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

@app.route('/api/model-info', methods=['GET'])
def model_info():
    try:
        # Load test data for metrics
        df = pd.read_csv("adfa_generated.csv")
        X_raw = df["sequence"]
        y = df["label"].values
        
        X_tfidf = vectorizer.transform(X_raw).toarray()
        X_selected = selector.transform(X_tfidf)
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.2, random_state=42
        )
        
        # Get predictions
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
        
        from sklearn.metrics import confusion_matrix, classification_report
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        return jsonify({
            'accuracy': round(accuracy * 100, 2),
            'loss': round(loss, 4),
            'confusion_matrix': cm.tolist(),
            'total_samples': len(X_selected),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'features': len(feature_names),
            'precision': round(report['weighted avg']['precision'], 4),
            'recall': round(report['weighted avg']['recall'], 4),
            'f1_score': round(report['weighted avg']['f1-score'], 4)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/example-sequences', methods=['GET'])
def example_sequences():
    """Return example syscall sequences"""
    df = pd.read_csv("adfa_generated.csv")
    normal = df[df['label'] == 0]['sequence'].iloc[0]
    attack = df[df['label'] == 1]['sequence'].iloc[0]
    
    return jsonify({
        'normal': normal,
        'attack': attack
    })

@app.route('/api/graphs/training-history', methods=['GET'])
def training_history():
    """Return training history for visualization"""
    try:
        if os.path.exists('training_history.json'):
            with open('training_history.json', 'r') as f:
                history = json.load(f)
            return jsonify(history)
        else:
            return jsonify({
                'loss': [],
                'accuracy': [],
                'val_loss': [],
                'val_accuracy': []
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/graphs/roc-curve', methods=['GET'])
def roc_curve_data():
    """Generate and return ROC curve data"""
    try:
        df = pd.read_csv("adfa_generated.csv")
        X_raw = df["sequence"]
        y = df["label"].values
        
        X_tfidf = vectorizer.transform(X_raw).toarray()
        X_selected = selector.transform(X_tfidf)
        
        from sklearn.model_selection import train_test_split
        _, X_test, _, y_test = train_test_split(
            X_selected, y, test_size=0.2, random_state=42
        )
        
        # Get predictions
        y_pred_probs = model.predict(X_test, verbose=0)
        y_pred_prob_positive = y_pred_probs[:, 1]
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob_positive)
        roc_auc = auc(fpr, tpr)
        
        return jsonify({
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'auc': float(roc_auc)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/graphs/precision-recall', methods=['GET'])
def precision_recall_data():
    """Generate and return Precision-Recall curve data"""
    try:
        df = pd.read_csv("adfa_generated.csv")
        X_raw = df["sequence"]
        y = df["label"].values
        
        X_tfidf = vectorizer.transform(X_raw).toarray()
        X_selected = selector.transform(X_tfidf)
        
        from sklearn.model_selection import train_test_split
        _, X_test, _, y_test = train_test_split(
            X_selected, y, test_size=0.2, random_state=42
        )
        
        # Get predictions
        y_pred_probs = model.predict(X_test, verbose=0)
        y_pred_prob_positive = y_pred_probs[:, 1]
        
        # Calculate Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_prob_positive)
        avg_precision = average_precision_score(y_test, y_pred_prob_positive)
        
        return jsonify({
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'avg_precision': float(avg_precision)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("[INFO] Loading models...")
    load_models()
    print("[INFO] Starting Flask app...")
    app.run(debug=False, port=5001)
