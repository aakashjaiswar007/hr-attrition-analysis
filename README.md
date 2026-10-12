# 👥 HR Attrition Intelligence System

> An end-to-end Machine Learning web app to predict and prevent 
> employee attrition using IBM HR Analytics data.

🔗 **Live App:** [Click here to open][(https://hr-attrition-analysis-intelligence.streamlit.app/)]

---

## 📌 Project Overview

This project analyzes 1,470 IBM employee records to identify the key 
drivers of employee attrition and predict which employees are at 
risk of leaving — using Machine Learning and an interactive web dashboard.

---

## 🔍 Key Findings

| Finding | Number |
|---------|--------|
| Total Employees Analyzed | 1,470 |
| Overall Attrition Rate | 16.1% |
| Model Accuracy | 87.07% |
| Avg Monthly Income | $6,503 |
| Avg Tenure | 7.0 years |
| Top Attrition Driver | Overtime + Low Salary |

---

## 🚀 Features

- ✅ Data cleaning and preprocessing pipeline
- ✅ Exploratory Data Analysis (EDA)
- ✅ Random Forest ML model with 87% accuracy
- ✅ Interactive dashboard with department filters
- ✅ Real-time employee attrition risk predictor
- ✅ 4 dynamic Plotly charts
- ✅ Deployed live on Streamlit Cloud

---

## 📊 Dashboard Sections

| Section | Description |
|---------|-------------|
| KPI Cards | Total employees, attrition rate, avg salary, tenure |
| Attrition by Department | Which department loses most employees |
| Income vs Attrition | Salary difference between stayed vs left |
| Overtime Impact | How overtime affects attrition |
| Top Attrition Drivers | Feature importance from Random Forest |
| Risk Predictor | Enter employee details → get risk score |

---

## 🔧 Tech Stack

| Area | Tools |
|------|-------|
| Data Cleaning | Python, Pandas, NumPy |
| Machine Learning | Scikit-learn, Random Forest |
| Visualization | Plotly, Matplotlib, Seaborn |
| Web App | Streamlit |
| Deployment | Streamlit Cloud |

---

## 📁 Project Structure
hr-attrition-analysis/
│
├── app.py # Main Streamlit dashboard
├── model.py # ML model training script
├── data_cleaning.py # Data cleaning functions
├── requirements.txt # Python dependencies
├── model.pkl # Saved trained model
└── README.md # Project documentation 



---

## ⚙️ How to Run Locally

```bash
# Clone the repo
git clone https://github.com/aakashjaiswar007/hr-attrition-analysis.git

# Go into folder
cd hr-attrition-analysis

# Install dependencies
pip install -r requirements.txt

# Train the model first
python model.py

# Run the app
python -m streamlit run app.py
```

---

## 📥 Dataset

**IBM HR Analytics Employee Attrition Dataset**
- Source: [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- Records: 1,470 employees
- Features: 35 columns

---

## 👤 Author

**Aakash Jaiswar**
- 📧 aakashjaiswar007@gmail.com
- 🔗 [LinkedIn](https://linkedin.com/in/aakash-jaiswar)
- 🐙 [GitHub](https://github.com/aakashjaiswar007)
