# House Price Prediction System

## Overview

This project is an interactive Machine Learning application developed using **Streamlit**. It provides a complete workflow for house price analysis, including data cleaning, visualization, classification, regression, and model comparison.

The application is designed as part of the NTI Graduation Project.

---

## Features

- Interactive Streamlit Dashboard
- Data Cleaning
- Exploratory Data Analysis (EDA)
- House Price Visualization
- House Price Classification using XGBoost
- House Price Prediction using Ridge Regression
- House Price Prediction using XGBoost Regression
- Model Performance Comparison
- Feature Importance Analysis
- Confusion Matrix
- Regression Metrics
- Interactive Prediction Interface

---

## Project Structure

```text
NTI_Graduation_Project/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│
├── pages/
│   ├── Home.py
│   ├── Data_Cleaning.py
│   ├── Data_Visualization.py
│   ├── Classification.py
│   ├── Regression.py
│   ├── XGBoost.py
│   ├── Model_Comparison.py
│   └── Contributors.py
│
├── services/
│   ├── data_cleaning.py
│   ├── data_visualization.py
│   ├── classification/
│   └── regression/
│
└── models/
```

---

## Machine Learning Models

### Classification

- XGBoost Classifier

### Regression

- Ridge Regression
- XGBoost Regressor

---

## Dataset

Dataset:

```
kc_house_data.csv
```

Target Variable:

```
price
```

---

## Technologies

- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost

---

## Installation

Clone the repository

```bash
git clone <repository-url>
```

Move to the project directory

```bash
cd NTI_Graduation_Project
```

Create a virtual environment

Windows

```bash
python -m venv .venv
```

Activate the virtual environment

PowerShell

```bash
.venv\Scripts\Activate.ps1
```

Command Prompt

```bash
.venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Application

Start the Streamlit application

```bash
streamlit run app.py
```

The application will be available at

```
http://localhost:8501
```

---

## Running with Ngrok

### Step 1

Start the Streamlit application

```bash
streamlit run app.py
```

### Step 2

Open another terminal.

### Step 3

Run Ngrok

```bash
ngrok http 8501
```

Ngrok will generate a public URL similar to

```
https://xxxxxxxx.ngrok-free.app
```

Share this URL to allow others to access the application.

---

## Project Workflow

1. Load Dataset
2. Clean Data
3. Visualize Data
4. Train Classification Model
5. Train Regression Models
6. Compare Models
7. Predict House Prices

---

## Contributors

- Abdelrahman Elhelbawy
- Abanob Nabil
- Marial Michel
- Mariam Ali
- Eman Ahmed
- Yomna Emad

---

## License

This project was developed for educational purposes as part of the NTI Graduation Project.