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

## 🛠️ Technologies
- **Python 3.x**
- **PyTorch** (Transformer implementation)
- **Pandas, NumPy** (data handling)
- **Scikit-learn** (scaling, evaluation)
- **Matplotlib** (visualization)
- **TQDM** (progress tracking)

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

## 🔎 File Explanations

### `load_data.py`
- Reads the raw `.txt` dataset file  
- Converts data into a **Pandas DataFrame**  
- Handles missing values  
- Ensures proper **time-indexed format** for downstream processing  

---

### `preprocessing.py`
- Resamples **minute-level measurements** into **hourly averages**  
- Normalizes values using techniques such as **MinMaxScaler**  
- Creates **sliding windows**:  
  - Input: 24-hour consumption history  
  - Target: 1-hour forecast (25th hour)  

---

### `model_transformer.py`
- Implements a **PyTorch Transformer Encoder**  
- Input: sequence of 24 hourly values  
- Output: predicted consumption for the **next hour**  
- Includes **positional encoding** to preserve temporal order  

---

### `train.py`
- Defines the **training loop** with:  
  - Loss function: **MSE** or **MAE**  
  - Optimizer: **Adam**  
  - Learning rate scheduling  
- Saves trained **model checkpoints** for later evaluation  

---

### `test_and_plot.py`
- Loads the trained model  
- Evaluates performance on the **test set**  
- Generates plots:  
  - **Actual vs Predicted consumption**  
  - **Error distribution** for diagnostic analysis  

---

## 📦 Installation
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
**Test FastAPI Requests**

```bash

python test_requests.py
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