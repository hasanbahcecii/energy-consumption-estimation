# ⚡ Energy Consumption Forecasting with Transformers

## 📌 Project Overview
Energy consumption reflects the electricity usage behavior of households, cities, and countries. Accurate forecasting is critical for **energy planning, resource allocation, and grid stability**.  

This project aims to **predict the next hour’s household energy consumption** based on the past 24 hours of usage data. Unlike classical time-series models (e.g., ARIMA, LSTM), we leverage **Transformer architectures** for improved sequence modeling and forecasting.

---

## 📂 Dataset
- **Source**: [UCI - Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)
- **Frequency**: Original data is recorded at **minute-level granularity**  
- **Preprocessing**: Convert minute-level data into **hourly averages**  
- **Objective**:  
  - Input: 24 hours of energy consumption  
  - Output: Forecast the **25th hour** (next hour’s consumption)

---

## ⚙️ Technologies
- **PyTorch** → Transformer model
- **FastAPI** → Web service (REST API)
- **Streamlit** → User interface
- **Python Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn, torch, fastapi, uvicorn, streamlit, requests

---

## 📑 Project Structure

```bash
energy-transformer/
│
├── load_data.py          # Load raw dataset into memory
├── preprocessing.py      # Clean, resample, and prepare hourly data
├── model_transformer.py  # Define Transformer-based forecasting model
├── train.py              # Training loop, optimizer, loss function
├── test_and_plot.py      # Evaluate model and visualize predictions
├── main_api.py             # FastAPI service
├── test_requests.py        # FastAPI request testing
├── app_streamlit.py        # Streamlit web app
├── requirements.txt        # Dependency list
├── .gitignore              # Excludes unnecessary files
└── README.md             # Project documentation
```

---

## 📦 Installation
### 1. Environment Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 📊 Example Workflow

**Load Data**
```bash

    python load_data.py
```
**Preprocess Data**
```bash

    python preprocessing.py
```
**Train Model**
```bash

    python train.py
```
**Test & Visualize**
```bash

    python test_and_plot.py
```
**Run FastAPI Service**

```bash

uvicorn main_api:app --reload
```

- Endpoint: http://127.0.0.1:8000/predict/

**Test FastAPI Requests**

```bash

python test_api.py
```
**Launch Streamlit App**

```bash

streamlit run app_streamlit.py
```

---

## 🚀 Future Improvements

- Extend forecasting horizon (multi-step prediction)

- Compare Transformer with LSTM/GRU baselines

- Hyperparameter tuning for better accuracy


---

## 📜 License

This project is for educational and research purposes.