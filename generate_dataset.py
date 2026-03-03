import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic ADFA-LD style dataset
num_samples = 2000  # Increased from 1000
sequence_length = 60  # Increased from 50

data = []

# Generate normal traffic (label 0)
for i in range(num_samples // 2):
    # Normal sequences: typical syscall patterns with limited range
    sequence = np.random.choice(range(10, 80), size=sequence_length, p=None)
    # Add some repeating patterns for normal traffic
    pattern_indices = np.random.choice(range(sequence_length), size=8, replace=False)
    for idx in pattern_indices:
        sequence[idx] = np.random.choice(range(10, 30))  # Common syscalls
    data.append({
        'sequence': ' '.join(map(str, sequence)),
        'label': 0
    })

# Generate attack traffic (label 1)
for i in range(num_samples // 2):
    # Attack sequences: include many anomalous syscall patterns
    sequence = np.random.choice(range(1, 150), size=sequence_length, p=None)
    # Add many distinct high-value patterns for attacks
    anomaly_count = np.random.randint(8, 16)  # More anomalies
    anomaly_indices = np.random.choice(range(sequence_length), size=anomaly_count, replace=False)
    for idx in anomaly_indices:
        sequence[idx] = np.random.choice(range(200, 350))  # Very distinct syscalls
    data.append({
        'sequence': ' '.join(map(str, sequence)),
        'label': 1
    })

# Create DataFrame and shuffle
df = pd.DataFrame(data)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to CSV
df.to_csv('adfa_generated.csv', index=False)

print(f"✓ Generated dataset with {len(df)} samples")
print(f"  - Normal traffic (label 0): {len(df[df['label'] == 0])} samples")
print(f"  - Attack traffic (label 1): {len(df[df['label'] == 1])} samples")
print(f"\nDataset saved to: adfa_generated.csv")
print(f"\nFirst 5 rows:")
print(df.head())
