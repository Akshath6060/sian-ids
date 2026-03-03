# Explainable IDS System

An advanced Intrusion Detection System with AI explainability using LIME and SHAP.

## Quick Start

```bash
# Clone
git clone <your-repo-url>
cd sian-ids

# Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run
python generate_dataset.py  # Generate data
python ids.py              # Train model
python app.py              # Start web app (http://127.0.0.1:5001)
```

## Features

✅ **81.25% Accuracy** on attack detection
✅ **LIME & SHAP** explainability 
✅ **Web Dashboard** for predictions
✅ **Real-time explanations** for model decisions

## Model Performance

- **Accuracy**: 81.25%
- **Precision**: 84.75%
- **Recall**: 81.25%
- **F1-Score**: 80.82%

See [README.md](README.md) for full documentation.
