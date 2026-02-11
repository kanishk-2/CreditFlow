# 📁 CreditFlow: Loan Approval System

**CreditFlow** is a production-grade Machine Learning application designed to automate credit risk assessment. By leveraging historical financial data and demographic profiles, it predicts loan eligibility with high precision using a multi-model comparative approach.

---

### 🚀 Overview
In the modern fintech landscape, speed and accuracy in credit decisions are vital. **CreditFlow** bridges the gap between raw financial data and actionable insights through a robust Scikit-learn pipeline and an interactive Streamlit interface.

---

### 🛠️ Technical Architecture
The system follows a strict ETL (Extract, Transform, Load) and ML lifecycle to ensure data integrity and model reliability.



#### 1. Data Intelligence & EDA
* **Feature Engineering:** Handling categorical variables via `OneHotEncoder` (dropping the first column to avoid the **Dummy Variable Trap**) and `LabelEncoder`.
* **Statistical Imputation:** Filling numerical gaps with **Mean** and categorical gaps with **Most Frequent** strategies.
* **Visualization:** Correlation heatmaps, distribution plots, and class balance analysis using `Seaborn` and `Matplotlib`.



#### 2. The Modeling Engine
While the application focuses on deployment stability, the development notebook evaluates three distinct algorithms to find the best fit:
* **Logistic Regression:** Baseline linear classification for interpretability.
* **K-Nearest Neighbors (KNN):** Pattern-based classification ($k=9$).
* **Gaussian Naive Bayes:** Probabilistic approach for feature independence.

#### 3. Production Safety
* **Standardization:** Feature scaling using `StandardScaler` to ensure distance-based models (like KNN) perform optimally.
* **Pipeline Integrity:** Strict feature alignment to handle unseen categories during real-time inference, preventing application crashes.

---

### 📊 Model Performance Matrix

| Model | Accuracy | Precision | Recall |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | High | Optimized | Balanced |
| **K-Nearest Neighbors** | Competitive | High | Sensitivity-focused |
| **Naive Bayes** | Robust | Fast | Probabilistic |

---

### 💻 Tech Stack
* **Language:** Python 3.x
* **Frontend:** Streamlit (Web UI)
* **Data Science:** Pandas, NumPy
* **Visualization:** Seaborn, Matplotlib
* **Machine Learning:** Scikit-Learn

---

### 📂 Project Structure
```bash
├── loan_approval_data.csv      # Dataset
├── CrediFlow-system.ipynb.     # EDA & Model Selection
├── app.py                      # Main Streamlit Application
├── requirements.txt            # Dependencies
└── README.md                   # Documentation
