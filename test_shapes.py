import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
import numpy as np

# Load data
df = pd.read_csv('adfa_generated.csv')
X_raw = df['sequence']
y = df['label'].values

# Create vectorizer
vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(2, 2), max_features=150)
X_tfidf = vectorizer.fit_transform(X_raw).toarray()

# Create selector
selector = SelectKBest(chi2, k=150)
X_selected = selector.fit_transform(X_tfidf, y)

print('TF-IDF shape:', X_tfidf.shape)
print('Selected shape:', X_selected.shape)

# Test with one sample
test_sequence = X_raw.iloc[0]
print('Test sequence (first 100 chars):', test_sequence[:100])

X_test_tfidf = vectorizer.transform([test_sequence]).toarray()
print('Test TF-IDF shape:', X_test_tfidf.shape)

X_test_selected = selector.transform(X_test_tfidf)
print('Test selected shape:', X_test_selected.shape)
