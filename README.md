# Machine Learning-Based Detection Of Privilege Escalation And Backdoor Persistence Using Host-Based Security Monitoring.

## Project Overview

This project investigates the use of machine learning to detect privilege escalation and backdoor persistence from Windows host-based security event logs.

The project focuses on identifying suspicious activity recorded in Windows security events and evaluating whether machine learning models can distinguish malicious behavior from normal system activity.

The implementation uses the TON_IoT Windows 7 Security Events dataset, which contains labeled Windows security event data, including events associated with backdoor activity.

## Dataset

The project uses the **TON_IoT Windows 7 Security Events dataset**.

The dataset contains Windows security event records that can be used to investigate and detect different types of malicious activity.

The raw dataset is not included in this repository.

## Methodology

The project follows a machine-learning-based security monitoring workflow:

**Windows Security Events → Data Preprocessing → Feature Selection → SMOTE → Model Training → Detection → Evaluation → Visualization**

## Data Preprocessing

The security event data was prepared for machine learning through:

- Handling missing values
- Removing duplicate records
- Removing constant and highly correlated features
- Selecting relevant features
- Encoding categorical information where required
- Addressing class imbalance using **SMOTE**

The original feature set was reduced from **192 features to 27 relevant features** after preprocessing and feature selection.

## Machine Learning Models

### Random Forest

Random Forest was used as the primary supervised machine learning model. It was trained using labeled security event data to classify activity based on the extracted features.

Feature importance was also analyzed to identify the features that contributed most to the model's predictions.

### Isolation Forest

Isolation Forest was used as an unsupervised anomaly detection approach for comparison.

It identifies observations that differ from the expected patterns within the security event data.

## Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- 5-Fold Cross-Validation

Hyperparameter optimization was also performed using **GridSearchCV**.

## Security Dashboard

A web-based security monitoring dashboard was developed using **Dash and Plotly**.

The dashboard presents the detection results and provides visualizations including:

- Detection results
- Event classifications
- Model confidence
- Feature importance
- Recent security events
- Detection statistics

## Technologies

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **imbalanced-learn**
- **Matplotlib**
- **Seaborn**
- **Dash**
- **Plotly**

## Academic Project

**Bachelor's Thesis — BSc in Cybersecurity**

German University of Technology in Oman (GUtech)
