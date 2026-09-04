# Fraud-Spike Detector for Digital Payments

## Overview

The **Fraud-Spike Detector** is a machine learning project designed to identify potentially fraudulent digital payment transactions.

The system analyzes historical transaction data and predicts the probability that a transaction is fraudulent. The project focuses on the challenges of real-world fraud detection, including highly imbalanced data, realistic time-based evaluation, feature engineering, fraud activity spikes, and business-aware decision making.

Instead of relying only on accuracy, the project evaluates the model using metrics such as **Precision, Recall, PR-AUC, and ROC-AUC**.

The final system uses an **XGBoost Classifier** and selects the fraud detection threshold based on expected business cost.

---

# Problem Statement

Digital payment platforms process a large number of transactions every day. Manually checking every transaction for fraud is not practical.

The objective of this project is to build a machine learning system that can:

* Analyze transaction data
* Predict the probability of fraud
* Identify suspicious transactions
* Reduce missed fraudulent transactions
* Minimize unnecessary blocking of legitimate transactions
* Select a fraud detection threshold based on business cost

---

# Dataset

This project uses the **Credit Card Fraud Detection Dataset**.

The dataset contains approximately:

* **284,807 transactions**
* **492 fraudulent transactions**
* **28 anonymized transaction features**
* Transaction time
* Transaction amount

The target variable is:

* `Class = 0` → Legitimate transaction
* `Class = 1` → Fraudulent transaction

---

# Major Challenge: Class Imbalance

One of the main challenges in fraud detection is that fraudulent transactions are extremely rare.

In this dataset, fraud represents only approximately **0.17%** of all transactions.

Because of this, accuracy alone is not a useful metric.

For example, a model that predicts every transaction as legitimate could achieve very high accuracy while detecting no fraud at all.

Therefore, this project focuses on:

* Precision
* Recall
* PR-AUC
* ROC-AUC
* Business Cost

---

# Project Workflow

```text
Transaction Dataset
        ↓
Data Exploration
        ↓
Feature Engineering
        ↓
Time-Based Train/Test Split
        ↓
XGBoost Model Training
        ↓
Fraud Probability Prediction
        ↓
Model Evaluation
        ↓
Cost-Based Threshold Optimization
        ↓
Final Fraud Detection
```

---

# Feature Engineering

The dataset already contains anonymized features from `V1` to `V28`.

Additional features were created to improve the model.

## 1. Log Transaction Amount

Transaction amounts can vary significantly.

For example:

```text
₹10
₹500
₹10,000
₹100,000
```

To reduce the effect of extremely large values, the transaction amount is transformed using:

```text
log(1 + Amount)
```

---

## 2. Hour of Transaction

The transaction time is used to derive the approximate hour of the day.

This allows the model to identify whether transaction behavior changes depending on the time of day.

---

# Fraud Spike Experiment

An important experiment in this project involved creating **burst features**.

The idea was that fraudulent activity may sometimes occur as a sudden spike of transactions.

For example:

```text
Transaction 1 → 12:00:01
Transaction 2 → 12:00:05
Transaction 3 → 12:00:08
Transaction 4 → 12:00:10
```

To capture this behavior, features were created for different time windows:

* Previous 60 seconds
* Previous 300 seconds
* Previous 3600 seconds

The features included information such as:

* Number of recent transactions
* Total transaction amount
* Transaction activity in recent time windows

---

# Challenge with Burst Features

The dataset does not contain:

* Card ID
* Customer ID
* Merchant ID

Because of this limitation, transaction spikes could not be calculated for a specific customer or card.

For example, ideally the system should answer:

> How many transactions did this particular card perform in the previous five minutes?

However, due to the lack of card or customer identifiers, the burst features could only represent more general transaction activity.

The experiment showed that the burst features did not improve model performance.

Approximately:

```text
Base Features PR-AUC       = 0.814
Base + Burst Features      = 0.801
```

Therefore, the burst features were excluded from the final model.

This experiment demonstrates the importance of validating feature engineering ideas instead of assuming that every new feature will improve machine learning performance.

---

# Model

The project uses an **XGBoost Classifier**.

XGBoost is a gradient boosting algorithm that combines multiple decision trees to make predictions.

The model receives transaction features and outputs a probability.

For example:

```text
Fraud Probability = 0.03 → Low Risk

Fraud Probability = 0.52 → Medium Risk

Fraud Probability = 0.92 → High Risk
```

Because fraud cases are rare, the model also accounts for class imbalance by giving additional importance to the minority fraud class during training.

---

# Time-Based Train/Test Split

Instead of randomly splitting the dataset, this project uses a chronological time-based split.

```text
First 70% of transactions
        ↓
Training Data

Later 30% of transactions
        ↓
Testing Data
```

This approach is more realistic for fraud detection.

In a real-world system, the model should learn from past transactions and predict future transactions.

A random split could mix future and past transactions, potentially creating unrealistic evaluation results or information leakage.

---

# Model Evaluation

The model is evaluated using the following metrics.

## Precision

Precision answers:

> Out of all transactions predicted as fraud, how many were actually fraudulent?

High precision helps reduce unnecessary fraud alerts.

---

## Recall

Recall answers:

> Out of all actual fraudulent transactions, how many did the model successfully detect?

High recall helps reduce missed fraud.

---

## PR-AUC

PR-AUC measures the relationship between precision and recall across different thresholds.

It is especially useful for highly imbalanced datasets such as fraud detection.

---

## ROC-AUC

ROC-AUC measures how well the model separates fraudulent and legitimate transactions across different classification thresholds.

---

# Cost-Based Threshold Optimization

A machine learning model produces a fraud probability, but a threshold is required to decide whether a transaction should be classified as fraud.

A common approach is:

```text
Probability ≥ 0.5 → Fraud
Probability < 0.5 → Legitimate
```

However, this project uses a more business-aware approach.

Two types of errors are considered.

## False Negative

A fraudulent transaction is incorrectly predicted as legitimate.

Possible consequence:

```text
Fraud is missed
Financial loss may occur
```

---

## False Positive

A legitimate transaction is incorrectly flagged as fraud.

Possible consequence:

```text
Customer inconvenience
Manual review cost
Potential loss of customer trust
```

The project assumes a cost of approximately **₹50** for each false positive.

Multiple probability thresholds are tested, and the threshold that minimizes the expected total business cost is selected.

The cost-optimal threshold was approximately:

```text
0.865
```

---

# Results

The final model achieved approximately:

| Metric                 |    Result |
| ---------------------- | --------: |
| PR-AUC                 | **0.814** |
| ROC-AUC                | **0.986** |
| Precision              | **93.3%** |
| Recall                 | **76.9%** |
| False Positives        |     **6** |
| Cost-Optimal Threshold | **0.865** |

The model was evaluated on approximately **85,439 transactions**, including **108 fraudulent transactions**.

The project estimated an approximately **79% reduction in fraud and review-related cost** compared with a baseline approach of flagging no transactions.

---

# Key Challenges

The major challenges faced during this project were:

## 1. Highly Imbalanced Data

Fraud cases represented only a very small percentage of the dataset.

This made accuracy unreliable and required the use of more appropriate metrics.

---

## 2. Limited Transaction Information

The dataset did not contain customer IDs, card IDs, or merchant IDs.

This limited the ability to calculate entity-specific fraud patterns and transaction velocity.

---

## 3. Burst Features Did Not Improve Performance

The fraud spike hypothesis was tested using multiple time windows.

However, the experimental burst features did not improve PR-AUC and were removed from the final model.

---

## 4. Avoiding Data Leakage

A time-based train/test split was used to ensure that the model learned from past transactions and was evaluated on future transactions.

---

## 5. Choosing the Correct Fraud Threshold

Using a default threshold of 0.5 may not be optimal for a business.

The project therefore selected a threshold based on expected fraud loss and false-positive costs.

---

# Project Structure

```text
fraud-spike-detector/
│
├── notebooks/
│   └── fraud_spike_detection.ipynb
│
├── src/
│   ├── features.py
│   ├── train.py
│   └── evaluate.py
│
├── outputs/
│   ├── trained_model
│   ├── metrics
│   └── evaluation_results
│
├── README.md
│
└── requirements.txt
```

---

# Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib
* Jupyter Notebook

---

# How to Run the Project

## 1. Clone the Repository

```bash
git clone <repository-url>
```

## 2. Move into the Project Directory

```bash
cd fraud-spike-detector
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Run the Notebook

```bash
jupyter notebook
```

Then open:

```text
notebooks/fraud_spike_detection.ipynb
```

---

# Future Improvements

The project could be improved by using additional transaction information such as:

* Card ID
* Customer ID
* Merchant ID
* Geographic location
* Device information
* IP address
* Merchant category

With this information, more powerful features could be created, such as:

* Transactions per card in the last 5 minutes
* Total spending per customer in the last hour
* Number of transactions per merchant
* Sudden changes in customer spending behavior
* Unusual geographic activity
* Device switching patterns

These features could improve the ability to detect real fraud spikes and suspicious customer behavior.

---

# Conclusion

The Fraud-Spike Detector demonstrates how machine learning can be applied to identify suspicious digital payment transactions.

The project highlights that fraud detection is not simply about achieving high accuracy. A useful fraud detection system must handle highly imbalanced data, avoid information leakage, evaluate meaningful metrics, test feature engineering ideas, and make decisions based on real business costs.

The final system uses an XGBoost model to generate fraud probabilities and applies a cost-based threshold to balance fraud detection against unnecessary customer disruption.

The project also demonstrates an important machine learning principle:

> A feature should not be included simply because it seems useful. It should be tested and evaluated based on measurable improvement.

This project provides a foundation for building more advanced fraud detection systems using richer transaction and customer information.
