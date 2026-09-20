# Employee Attrition Prediction

Machine learning project that predicts whether an employee is likely to leave a company, using the IBM HR Analytics dataset. Includes EDA, preprocessing, XGBoost modeling, SHAP explainability, and a Streamlit app for live predictions.

## Features

- Exploratory data analysis of attrition drivers
- Preprocessing pipeline (encoding, scaling, class imbalance handling)
- XGBoost classifier for attrition prediction
- SHAP explanations for global and per-employee feature impact
- Streamlit web app for interactive predictions

## Project Structure

```
employee-attrition-prediction/
├── app/
│   └── streamlit_app.py        # Streamlit UI
├── data/
│   ├── raw/                    # Original dataset
│   └── processed/              # Cleaned / encoded data
├── models/
│   ├── xgboost.pkl             # Trained model
│   └── shap_explainer.pkl      # SHAP explainer
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_Modeling.ipynb
│   └── 04_SHAP_Explainability.ipynb
├── reports/figures/            # Plots used in the analysis
├── src/
│   ├── data_preprocessing.py
│   ├── train.py
│   └── predict.py
├── requirements.txt
└── README.md
```

## Dataset

[IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) (`WA_Fn-UseC_-HR-Employee-Attrition.csv`): 1,470 employees, 35 features, target column `Attrition`.

## Setup

```bash
git clone https://github.com/Pratyushhh003/employee-attrition-prediction.git
cd employee-attrition-prediction
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

## Usage

Train the model:

```bash
python src/train.py
```

Run the app:

```bash
streamlit run app/streamlit_app.py
```



## Tech Stack

Python, pandas, NumPy, scikit-learn, XGBoost, SHAP, matplotlib, seaborn, Streamlit

## Author

**Pratyush Das**  **Registration Number 25225017** · [GitHub](https://github.com/Pratyushhh003)
**Morgan John**  **Registration Number 25225014** 
**Arpan Maiti**  **Registration Number 25225006** · 
