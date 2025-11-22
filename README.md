
# 🚗 Predictive Maintenance – End-to-End MLOps Project

This project implements a complete **Predictive Maintenance Pipeline** for engine condition classification using real-like sensor data.  
It includes **data preprocessing**, **feature engineering**, **model training**, **MLflow experiment tracking**, **Streamlit dashboard**, **unit tests**, and **CI/CD automation**.

---

## 📁 Project Structure

```

.
├── app/                       # Streamlit dashboard
│   └── app.py
├── data/
│   ├── raw/                   # Raw CSV dataset
│   └── processed/             # Processed dataset
├── models/                    # Saved best model (.joblib)
├── mlruns/                    # MLflow tracking data
├── scripts/
│   ├── preprocess.py          # Data cleaning + feature engineering
│   ├── train.py               # Training pipeline (RF + XGBoost)
│   └── check_performance.py   # Automated model quality validation
├── tests/
│   └── test_preprocess.py     # Unit tests for feature engineering
├── requirements.txt
├── debug_model.py
└── README.md

```

---

## 🧠 Project Overview

This predictive maintenance system uses engine sensor data to classify the **Engine Condition** as:

- **0 = Normal**
- **1 = Abnormal**

The pipeline includes:

### ✔ Data Preprocessing
- Handling missing values  
- Cleaning sensor columns  
- Adding engineered features:
  - `Engine_power = Engine rpm × Lub oil pressure`
  - `Temperature_difference = Coolant temp – Lub oil temp`

### ✔ Model Training
Two models are trained:
- **Random Forest**
- **XGBoost**

Metrics used:
- **Accuracy**
- **ROC-AUC**  
(ROC-AUC is used to select the best model)

### ✔ MLflow Tracking
All experiments are logged:
- parameters  
- metrics  
- models  
- artifacts  

The best model is stored as:
```

models/best_model.joblib

````

### ✔ Streamlit Dashboard
A clean UI that allows:
- adjusting sensor values  
- generating predictions  
- visualizing derived features  

---

## ▶️ How to Run the Project

### **1️⃣ Create a virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows
````

### **2️⃣ Install dependencies**

```bash
pip install -r requirements.txt
```

---

## 🧹 Run Preprocessing

```bash
python scripts/preprocess.py \
    --input data/raw/engine_data.csv \
    --output data/processed/processed.csv
```

---

## 🤖 Train Models

```bash
python scripts/train.py \
    --input data/processed/processed.csv \
    --target "Engine Condition"
```

This generates:

* `models/best_model.joblib`
* MLflow experiment logs in `mlruns/`

---

## 🧪 Run Tests

```bash
pytest -v
```

Includes tests for:

* feature creation
* missing column handling
* datatype validation
* edge cases

---

## 🖥 Launch the Streamlit Dashboard

```bash
streamlit run app/app.py
```

Features:

* interactive sensor sliders
* live prediction
* confidence score
* auto-calculation of engine power and temp difference

---

## 🔧 CI/CD Pipeline (GitHub Actions)

A workflow automatically:

* executes preprocessing
* trains the model
* validates performance (ROC-AUC threshold)
* uploads:

  * the trained model
  * MLflow tracking folder
  * performance report

Located at:

```
.github/workflows/train.yml
```

---

## 📈 Example Results

* **Best model:** RandomForest
* **Accuracy:** ~65–68%
* **ROC-AUC:** ~0.67
* **Interpretation:**
  Dataset is simple and synthetic → performance reasonable.

---

## 🏁 Conclusion

This project demonstrates:

* End-to-end ML pipeline
* MLOps practices (MLflow + CI/CD)
* Model training & evaluation
* Interactive dashboard
* Testing and reproducibility

It is suitable for **Data Engineer**, **AI Engineer**, and **MLOps** portfolios.

