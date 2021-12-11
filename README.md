# Logistic Regression and Adaboost for Classification

In this assignment of CSE472 course we implement adaboost algorithm with
logistic regression as weak learner to predict the binary labels of given
datasets.

## Report of Logistic Regression

### Telco dataset

| Performance measure       | Training  | Test  |
| :---                      | :---:     | :---: |
| True positive rate        | 56%       | 52%   |
| True negative rate        | 90%       | 90%   |
| Positive predictive value | 67%       | 65%   |
| False discovery rate      | 33%       | 36%   |
| F1 score                  | 61%       | 57%   |

### Adult dataset

| Performance measure       | Training  | Test  |
| :---                      | :---:     | :---: |
| True positive rate        | 00%       | 00%   |
| True negative rate        | 01%       | 01%   |
| Positive predictive value | nan       | nan   |
| False discovery rate      | nan       | nan   |
| F1 score                  | 00%       | 00%   |

### Credit Card Fraud dataset

| Performance measure       | Training  | Test  |
| :---                      | :---:     | :---: |
| True positive rate        | 94%       | 93%   |
| True negative rate        | 41%       | 41%   |
| Positive predictive value | 03%       | 03%   |
| False discovery rate      | 96%       | 96%   |
| F1 score                  |  7%       | 07%   |

## Report of Adaboost

### Telco dataset

| Number of boosting rounds | Training  | Test  |
| :---:                     | :---:     | :---: |
| 5                         | 79%       | 78%   |
| 10                        | 81%       | 79%   |
| 15                        | 80%       | 79%   |

### Adult dataset

| Number of boosting rounds | Training  | Test  |
| :---:                     | :---:     | :---: |
| 5                         | 99%       | 99%   |
| 10                        | 99%       | 99%   |
| 15                        | 99%       | 99%   |

### Credit Card Fraud dataset

| Number of boosting rounds | Training  | Test  |
| :---:                     | :---:     | :---: |
| 5                         | 51%       | 51%   |
| 10                        | 98%       | 98%   |
| 15                        | 96%       | 96%   |

