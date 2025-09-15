# Financial Fraud Detection with Machine Learning

## Overview

This project develops a machine learning model to detect fraudulent financial transactions from a synthetic dataset that simulates mobile money transactions. The goal is to build a highly accurate and reliable system that can identify fraudulent behavior in real-time, moving beyond simple rule-based methods. The final model, an XGBoost Classifier, successfully identifies **98% of fraudulent transactions** in the test set.

-----

## Business Problem

Fraudulent transactions represent a significant financial risk and can damage customer trust. [cite\_start]The fraudulent behavior in this dataset aims to profit by taking control of customer accounts and emptying the funds by transferring them to another account and then cashing out[cite: 8]. [cite\_start]The existing prevention system relies on a static, simplistic rule of flagging transfers over 200,000 in a single transaction[cite: 10]. This project aims to build a more intelligent, data-driven system that can identify complex fraudulent patterns with high precision and recall.

-----

## Dataset

The project uses a synthetic dataset called **PaySim**, which was generated using a simulator based on aggregated real-world financial transaction data.

  * **Source**: [Kaggle - Synthetic Financial Datasets for Fraud Detection](https://www.kaggle.com/datasets/ealaxi/paysim1)
  * [cite\_start]**Content**: The dataset contains over 6.3 million transactions from a 30-day simulation[cite: 2].
  * [cite\_start]**Features**: Key features include transaction `type` (e.g., `CASH_OUT`, `TRANSFER`) [cite: 3][cite\_start], `amount`, initial and final balances for both the originator and recipient, and a target label `isFraud`[cite: 7].

-----

## Project Workflow

1.  **Exploratory Data Analysis (EDA)**: Investigated the dataset to identify key characteristics, such as the extreme class imbalance (only 0.129% of transactions are fraudulent) and the fact that fraud occurs exclusively in `TRANSFER` and `CASH_OUT` transaction types.
2.  **Feature Engineering**: Created new, powerful features (`errorBalanceOrig` and `errorBalanceDest`) to capture discrepancies in account balances after transactions, which proved to be highly predictive.
3.  **Model Training**: Addressed the extreme class imbalance by using the **SMOTE (Synthetic Minority Over-sampling Technique)** on the training data. This ensures the model learns the patterns of the rare fraud cases effectively.
4.  **Model Evaluation**: Trained and compared a baseline Logistic Regression model with an advanced **XGBoost Classifier**. The XGBoost model demonstrated vastly superior performance.
5.  **Interpretation**: Analyzed the feature importances from the trained model to identify the key drivers of fraud.

-----

## Results & Key Insights

The final XGBoost model achieved outstanding performance on the unseen test data:

  * **Recall**: **98%** (It successfully catches 98 out of every 100 fraudulent transactions).
  * **Precision**: **39%** (Significantly reduces false alarms compared to the baseline).
  * **AUC Score**: **0.99**

The most important factors the model uses to predict fraud are:

1.  **`type_TRANSFER`**: Whether the transaction is a transfer, the first step in the fraud pattern.
2.  **`errorBalanceOrig`**: Discrepancies in the sender's account balance.
3.  **Account Balances**: The initial and final balances of the accounts involved.

-----

## Proposed Solution

Based on the model's high performance, we recommend implementing a **real-time, three-tiered alert system**:

  * **High-Risk Score (\>0.9)**: Automatically **block** the transaction for review.
  * **Medium-Risk Score (0.5-0.9)**: Challenge the user with **2-Factor Authentication (2FA)**.
  * **Low-Risk Score (\<0.5)**: **Allow** the transaction to proceed without friction.

This system provides a robust defense against fraud while maintaining a smooth experience for legitimate users. Its effectiveness can be validated via A/B testing against the current system.

-----

## How to Use This Repository

### Prerequisites

  * Python 3.8+
  * Jupyter Notebook or JupyterLab

### Installation

1.  Clone the repository:
    ```bash
    [git clone https://github.com/Onkar-Ai/Financial-Fraud-Detection-with-Machine-Learning]
    cd fraud-detection-project
    ```
2.  Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Analysis

Open and run the `Fraud_Detection_Analysis.ipynb` notebook in Jupyter.

-----

## File Structure

```
├── Fraud_Detection_Analysis.ipynb    # Main notebook with all the analysis and modeling code.
├── paysim.csv                        # The dataset (note: large file).
├── requirements.txt                  # A list of required Python libraries.
└── README.md                         # You are here.
```

-----

To complete your repository, you should also create a `requirements.txt` file. Here is the content for it:

```txt
pandas
numpy
scikit-learn
xgboost
imblearn
seaborn
matplotlib
jupyter
```
