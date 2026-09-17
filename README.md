# Home Loan Eligibility Prediction

## Project Overview

The objective of this project is to predict whether a home loan application is likely to be approved or rejected.

The project uses data preprocessing, exploratory data analysis, feature engineering, and Logistic Regression to identify important factors that influence loan approval.

---

## Business Objective

The main objectives of this project are:

- Understand patterns in home loan application data.
- Handle missing values and categorical features.
- Perform feature engineering.
- Identify important factors affecting loan approval.
- Build a machine learning model to predict loan eligibility.

---

## Machine Learning Problem

This is a **Binary Classification Problem**.

Target Variable:

- `1` = Loan Approved
- `0` = Loan Rejected

Logistic Regression was used as the classification model.

---

## Dataset Overview

The dataset contains:

- **700 records**
- **13 original columns**

The original features include:

- Loan ID
- Gender
- Married
- Dependents
- Education
- Self Employed
- Applicant Income
- Coapplicant Income
- Loan Amount
- Loan Amount Term
- Credit History
- Property Area
- Loan Status

---

## Tools and Technologies

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## Project Workflow

### 1. Data Understanding

The dataset was examined using:

- `head()`
- `tail()`
- `shape`
- `columns`
- `dtypes`
- `info()`
- `describe()`

This helped understand the dataset structure, data types, distributions, and missing values.

---

## Missing Value Analysis

Missing values were found in:

- Gender
- Married
- Dependents
- Self Employed
- Loan Amount
- Loan Amount Term
- Credit History

Categorical missing values were handled using the **mode**.

Numerical missing values were handled using appropriate statistical values such as the **median**.

---

## Data Conversion

Several columns were converted into suitable formats.

The `Dependents` value `3+` was converted to `3`.

Credit History was converted into integer format.

The target variable `Loan_Status` was converted into numerical format:

- `Y` = 1
- `N` = 0

---

## Feature Engineering

New features were created to improve the analysis and prediction.

### Total Income

Applicant Income and Coapplicant Income were combined.

```python
data['TotalIncome'] = data['ApplicantIncome'] + data['CoapplicantIncome']
```

### Total Income Log

Log transformation was applied to Total Income to reduce skewness.

### Loan Amount Log

Log transformation was also applied to Loan Amount.

### EMI

EMI was calculated as:

```python
data['EMI'] = data['LoanAmount'] / data['Loan_Amount_Term']
```

### EMI to Income Ratio

This feature measures repayment burden compared with total income.

### Income to Loan Ratio

This feature compares total income with the requested loan amount.

### Family Size

Family size was created using the number of dependents.

```python
data['FamilySize'] = data['Dependents'] + 1
```

---

## Categorical Encoding

Categorical variables were converted into numerical format using one-hot encoding.

```python
pd.get_dummies()
```

This allowed the categorical features to be used by the machine learning model.

---

## Exploratory Data Analysis

### Loan Approval Distribution

The dataset contains:

- **526 Approved Applications**
- **174 Rejected Applications**

Approximately:

- **75% Approved**
- **25% Rejected**

---

### Credit History vs Loan Approval

Credit History was one of the strongest factors associated with loan approval.

Applicants with a good credit history had a much higher approval rate.

This indicates that previous credit behavior plays an important role in loan eligibility.

---

### Income vs Loan Approval

Approved applicants generally showed slightly higher income.

However, the difference between approved and rejected applicants was not very large.

This suggests that income alone does not determine loan approval.

---

### EMI to Income Ratio

Applicants with a higher EMI-to-income ratio showed slightly higher rejection chances.

This indicates that repayment burden can influence the loan decision.

---

### Family Size

Family size was analyzed to understand whether the number of dependents affects loan approval.

The results showed differences in approval rates across different family sizes.

---

### Marital Status

Married applicants showed a slightly higher loan approval rate compared with unmarried applicants.

---

### Property Area

Applicants from different property areas showed different approval patterns.

Semi-urban applicants showed relatively strong approval rates compared with some other property areas.

---

### Correlation Analysis

A correlation heatmap was created to understand relationships among the numerical features.

Credit History showed one of the strongest relationships with Loan Status.

Income-related engineered features also contributed to the analysis.

---

## Data Preprocessing

Before model training:

- Categorical variables were encoded.
- Unnecessary columns were removed.
- Numerical features were standardized using `StandardScaler`.
- The target variable was separated from the input features.

---

## Train-Test Split

The dataset was divided into:

- **80% Training Data**
- **20% Testing Data**

The training data was used to train the model.

The testing data was used to evaluate model performance.

---

## Machine Learning Model

### Logistic Regression

Logistic Regression was selected because the target variable contains two classes:

- Approved
- Rejected

The model was trained using:

```python
LogisticRegression(max_iter=1000)
```

---

## Model Performance

The Logistic Regression model achieved approximately:

| Metric | Result |
|---|---:|
| Accuracy | 90.7% |
| Precision for Approved Loans | 89% |
| Recall for Approved Loans | 100% |
| F1-Score for Approved Loans | 94% |

---

## Confusion Matrix

```text
[[18, 13],
 [0, 109]]
```

### Confusion Matrix Interpretation

- **109** approved applications were correctly predicted as approved.
- **18** rejected applications were correctly predicted as rejected.
- **13** rejected applications were incorrectly predicted as approved.
- **0** approved applications were incorrectly predicted as rejected.

The model performs very well in identifying approved applications.

However, the model has lower recall for rejected applications, meaning some risky applicants may be incorrectly predicted as approved.

---

## Key Insights

1. Credit History is one of the most important factors associated with loan approval.
2. Applicants with a good credit history have a much higher chance of approval.
3. Income contributes to loan eligibility, but income alone does not determine approval.
4. Higher EMI-to-income ratios may increase repayment burden and rejection risk.
5. Property area and marital status also show differences in approval patterns.
6. Feature engineering helped create more meaningful financial indicators.

---

## Business Recommendations

### 1. Give Strong Importance to Credit History

Credit History should remain an important factor while evaluating loan applications.

### 2. Monitor Repayment Burden

EMI-to-income ratio can be used to identify applicants who may face higher repayment pressure.

### 3. Use Multiple Factors Together

Loan approval decisions should not depend only on income.

Credit History, income, loan amount, repayment burden, and other applicant characteristics should be considered together.

### 4. Review High-Risk Predictions

Since some rejected applicants were predicted as approved, applications with uncertain or risky profiles should receive additional review.

---

## Conclusion

This project demonstrates how machine learning can be used to predict home loan eligibility.

The Logistic Regression model achieved approximately **90.7% accuracy**.

The model successfully predicted all approved applications in the test set, while some rejected applications were incorrectly predicted as approved.

The analysis showed that Credit History is one of the strongest factors associated with loan approval, while income, EMI burden, property area, and other applicant characteristics also contribute to the decision.

---

## Repository Structure

```text
Home-Loan-Eligibility-Prediction/
│
├── Home_Loan_Eligibility_Prediction.ipynb
├── realistic_home_loan_dataset_700.csv
├── README.md
└── requirements.txt
```

---

## How to Run the Project

1. Download or clone this repository.
2. Install the required Python libraries.
3. Open the Jupyter Notebook.
4. Keep `realistic_home_loan_dataset_700.csv` in the project folder.
5. Run all notebook cells from top to bottom.

---

## Author

**Barla Charan**

MBA – Systems with Business Analytics

