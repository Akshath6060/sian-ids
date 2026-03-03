# 🛡️ Explainable IDS System

An advanced Intrusion Detection System (IDS) with AI explainability using LIME and SHAP for detecting network attacks.

## Features

✨ **Machine Learning**
- TF-IDF Vectorization with 2-gram and 3-gram features
- Chi-Square feature selection
- Deep Neural Network classifier with Batch Normalization and Dropout
- **Accuracy: 81.25%** on synthetic ADFA-LD dataset

📊 **Explainability**
- **LIME**: Local Interpretable Model-agnostic Explanations
- **SHAP**: SHapley Additive exPlanations
- Feature importance visualization

🎯 **Performance Metrics**
- Precision: 84.75%
- Recall: 81.25%
- F1-Score: 80.82%

## Installation

### Requirements
- Python 3.12+
- TensorFlow/Keras
- scikit-learn
- LIME & SHAP
- Flask

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/sian-ids.git
cd sian-ids
```

2. **Create virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Generate dataset**
```bash
python generate_dataset.py
```

5. **Train the model** (optional, models are cached)
```bash
python ids.py
```

## Usage

### Web Interface

Start the Flask web application:
```bash
python app.py
```

Open browser and navigate to: **http://127.0.0.1:5001**

Features:
- Enter syscall sequences to get predictions
- View LIME feature importance
- See model performance metrics
- Analyze confusion matrix

### Command Line

Run the IDS training and evaluation:
```bash
python ids.py
```

## Project Structure

```
sian-ids/
├── ids.py                  # Main IDS implementation
├── app.py                  # Flask web application
├── generate_dataset.py     # Synthetic dataset generator
├── templates/
│   └── index.html         # Web UI
├── static/
│   ├── style.css          # Styling
│   └── script.js          # Frontend logic
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Model Architecture

```
Input (180 features)
    ↓
Dense(256) + BatchNorm + Dropout(0.3)
    ↓
Dense(128) + BatchNorm + Dropout(0.3)
    ↓
Dense(96) + BatchNorm + Dropout(0.2)
    ↓
Dense(64) + BatchNorm + Dropout(0.2)
    ↓
Dense(32) + Dropout(0.15)
    ↓
Output (2 classes: Normal/Attack)
```

## Configuration

### Feature Engineering
- **n-grams**: 2-grams and 3-grams
- **Max features**: 200
- **Selected features**: 180

### Model Training
- **Epochs**: 50
- **Batch size**: 16
- **Learning rate**: 0.0005
- **Early stopping**: patience=5

## Example

### Prediction via API

```bash
curl -X POST http://127.0.0.1:5001/api/predict \
  -H "Content-Type: application/json" \
  -d '{"sequence": "76 104 47 43 148 93 144 78 7 70 149 36 104 54 147 25 67 109 222 91 2 134 127 46 246 40 58 78 33 289"}'
```

Response:
```json
{
  "prediction": "Attack Traffic",
  "confidence": 81.47,
  "class": 1,
  "lime_features": [
    ["49 64 <= 0.00", 0.521],
    ["54 87 <= 0.00", 0.512],
    ...
  ]
}
```

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | 81.25% |
| Precision | 84.75% |
| Recall | 81.25% |
| F1-Score | 80.82% |
| Train Samples | 1,600 |
| Test Samples | 400 |

## Performance

- **LIME explanation time**: ~0.13 seconds
- **SHAP explanation time**: ~2.9 seconds
- **Model inference**: <5ms

## Future Improvements

- [ ] Real ADFA-LD dataset integration
- [ ] Ensemble methods (Random Forest, XGBoost)
- [ ] Real-time anomaly detection
- [ ] Docker containerization
- [ ] REST API authentication
- [ ] Database logging

## Technologies Used

- **ML/DL**: TensorFlow, Keras, scikit-learn
- **Explainability**: LIME, SHAP
- **Backend**: Flask
- **Frontend**: HTML5, CSS3, JavaScript
- **Data Processing**: NumPy, Pandas



## Author

Created with ❤️ 😌for Explainable AI in Network Security

## Contact

For questions or suggestions, please open an issue on GitHub.
