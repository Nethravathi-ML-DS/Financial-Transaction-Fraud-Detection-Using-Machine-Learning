# Financial-Transaction-Fraud-Detection-Using-Machine-Learning

## Executive Summary
This project builds a predictive machine learning model to detect fraudulent financial transactions using a dataset of 6.36+ million records. The goal is to identify high-risk `TRANSFER` and `CASH_OUT` events while minimizing false positives to maintain customer trust and regulatory compliance.

## Key Insights & Findings
* **Severe Class Imbalance**: Fraudulent transactions represent only **0.13%** of the dataset (8,213 out of 6,362,620 total transactions).
* **Transaction Type Specificity**: Fraud events occur exclusively within `TRANSFER` and `CASH_OUT` transaction types. All other transaction types (`PAYMENT`, `CASH_IN`, `DEBIT`) showed zero instances of fraud.
* **System Flagging Limitations**: The automated system flag (`isFlaggedFraud`) triggered only 16 times across 6.36M transactions, indicating a strong need for improved ML-driven detection logic.

## Project Structure
```text
├── data/                  # Dataset directory (or link to Kaggle/external source)
├── notebooks/             # Jupyter Notebooks containing EDA and Model Training
│   └── fraud_detection.ipynb
├── src/                   # Source code scripts (if modularized)
├── README.md              # Project documentation
└── requirements.txt       # Project dependencies


Dataset Overview
The dataset consists of 6,362,620 rows and 11 features:

step: Maps a unit of time in the real world (1 step = 1 hour).

type: Type of transaction (PAYMENT, TRANSFER, CASH_OUT, DEBIT, CASH_IN).

amount: Amount of the transaction in local currency.

nameOrig / nameDest: Customer ID initiating/receiving the transaction.

oldbalanceOrg / newbalanceOrig: Initial and new balances of the sender.

oldbalanceDest / newbalanceDest: Initial and new balances of the receiver.

isFraud: Target variable (1 = Fraud, 0 = Legitimate).

isFlaggedFraud: Internal system flag for transactions > 200,000 units.

Tech Stack & Dependencies
Language: Python 3.x

Data Processing & Analysis: pandas, numpy

Visualization: matplotlib, seaborn

Machine Learning: scikit-learn, XGBoost.

How to Run the Project
Clone the repository:

Bash
git clone [https://github.com/Nethravathi-ML-DS/Financial-Transaction-Fraud-Detection-Using-Machine-Learning.git]
cd fraud-detection-ml
Install required dependencies:

Bash
pip install -r requirements.txt
Run the Jupyter Notebook:

Bash
jupyter notebook notebooks/fraud_detection.ipynb


## 2. Refactor & Professionalize Your Jupyter Notebook

Before uploading your notebook, clean it up to show strong coding standards:

* **Use Markdown Headers inside the Notebook**: Divide your code into clear sections:
  1. *1. Environment Setup & Data Loading*
  2. *2. Exploratory Data Analysis (EDA)*
  3. *3. Data Cleaning & Feature Engineering*
  4. *4. Model Building & Evaluation*
  5. *5. Conclusion & Business Impact*
* **Add Code Comments**: Explain *why* you are running specific checks (e.g., `# Check for missing values to ensure data integrity`).
* **Clean Up Outdated Outputs**: Suppress unnecessary warnings and remove empty/failed code cells.
* **Save Notebook with Rendered Plots**: GitHub renders Jupyter Notebooks directly. Ensure all `matplotlib` and `seaborn` plots are generated and visible so recruiters can view them without running the notebook.

---

## 3. Include Recommended Project Files

To complete your repository setup:

* **`requirements.txt`**: List the libraries needed to run your code so reviewers can reproduce your environment:
  ```text
  pandas>=2.0.0
  numpy>=1.24.0
  matplotlib>=3.7.0
  seaborn>=0.12.0
  scikit-learn>=1.2.0
