import requests
import numpy as np

# load test dataset
X_test = np.load("X_test.npy")

# take one sample for test
sample_index = 0
sample = X_test[sample_index]  # örnek: (72, 7) boyutlu veri

# make it suitable for json (list of lists)
payload = {
    "sequence": sample.tolist()
}

# API url
url = "http://127.0.0.1:8000/predict"

# send request
try:
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        result = response.json()
        print(f"Predicted NO2: {result}")
    else:
        print(f"Error code: {response.status_code}")
        print(f"Error message: {response.text}")

except requests.exceptions.ConnectionError:
    print("The API server is not responding. Please, start FastAPI!")
