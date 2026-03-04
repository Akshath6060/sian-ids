import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic ADFA-LD style dataset
num_samples = 5000  # Increased from 2000 for better training
sequence_length = 80  # Increased from 60

data = []

# Generate NORMAL traffic (label 0) - Clean, predictable patterns
for i in range(num_samples // 2):
    # Normal sequences: very Limited syscall patterns (10-50)
    sequence = np.zeros(sequence_length, dtype=int)
    
    # Use common low-numbered syscalls for normal traffic
    common_syscalls = [1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 25]
    
    # Fill sequence with normal syscalls
    for j in range(sequence_length):
        if j % 3 == 0:  # Add repetitive patterns
            sequence[j] = np.random.choice([10, 11, 12, 13, 14])  # Most common
        elif j % 3 == 1:
            sequence[j] = np.random.choice([1, 2, 3, 4, 5, 20])
        else:
            sequence[j] = np.random.choice(common_syscalls)
    
    data.append({
        'sequence': ' '.join(map(str, sequence)),
        'label': 0
    })

# Generate ATTACK traffic (label 1) - Anomalous patterns
for i in range(num_samples // 2):
    # Attack sequences: Mix of normal with many anomalies
    sequence = np.zeros(sequence_length, dtype=int)
    
    # Start with some normal syscalls
    for j in range(sequence_length):
        if np.random.random() < 0.3:  # 30% normal syscalls
            sequence[j] = np.random.choice([1, 2, 3, 4, 5, 10, 11, 12, 13, 14])
        else:  # 70% anomalous syscalls
            sequence[j] = np.random.choice(range(100, 350))  # Very distinct attack syscalls
    
    # Add more high-value anomalies
    anomaly_count = np.random.randint(30, 50)  # More anomalies
    anomaly_indices = np.random.choice(range(sequence_length), size=anomaly_count, replace=False)
    for idx in anomaly_indices:
        sequence[idx] = np.random.choice(range(200, 400))  # Even more extreme
    
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
