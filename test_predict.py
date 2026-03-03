import requests
import json

# Test the predict API
url = "http://127.0.0.1:5001/api/predict"

# Sample syscall sequence
sequence = "76 104 47 43 148 93 144 78 7 70 149 36 104 54 147 25 67 109 222 91 2 134 127 46 246 40 58 78 33 289"

payload = {
    "sequence": sequence
}

print("Testing predict endpoint...")
print(f"URL: {url}")
print(f"Payload: {payload}")

try:
    response = requests.post(url, json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
    
    if response.status_code == 200:
        data = response.json()
        print("\nPrediction successful!")
        print(json.dumps(data, indent=2))
    else:
        print("\nError response:")
        print(response.text)
except Exception as e:
    print(f"Request error: {e}")
