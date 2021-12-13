# Logistic Regression and Adaboost for Classification

In this assignment of CSE472 course we implement adaboost algorithm with
logistic regression as weak learner to predict the binary labels of given
datasets.

# Datasets Used

- https://www.kaggle.com/blastchar/telco-customer-churn
- https://archive.ics.uci.edu/ml/datasets/adult
- https://www.kaggle.com/mlg-ulb/creditcardfraud

The first two datasets are saved in this repo. The third one is too large to
push to GitHub, so you'll need to download it yourself.

```bash
wget -O creditcard.zip https://www.kaggle.com/mlg-ulb/creditcardfraud/download
unzip creditcard.zip
mv creditcard.csv ./data/
```

## Report of Logistic Regression

### Telco dataset

| Performance measure       | Training  | Test  |
| :---                      | :---:     | :---: |
| Accuracy                  | 80%       | 79%   |
| True positive rate        | 53%       | 50%   |
| True negative rate        | 89%       | 89%   |
| Positive predictive value | 65%       | 63%   |
| False discovery rate      | 34%       | 36%   |
| F1 score                  | 58%       | 56%   |

### Adult dataset

| Performance measure       | Training  | Test  |
| :---                      | :---:     | :---: |
| Accuracy                  | 84%       | 84%   |
| True positive rate        | 58%       | 58%   |
| True negative rate        | 92%       | 92%   |
| Positive predictive value | 72%       | 71%   |
| False discovery rate      | 27%       | 28%   |
| F1 score                  | 64%       | 64%   |

### Credit Card Fraud dataset

| Performance measure       | Training  | Test  |
| :---                      | :---:     | :---: |
| Accuracy                  | 42%       | 42%   |
| True positive rate        | 94%       | 93%   |
| True negative rate        | 41%       | 41%   |
| Positive predictive value | 03%       | 03%   |
| False discovery rate      | 96%       | 96%   |
| F1 score                  | 07%       | 07%   |

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
| 5                         | 84%       | 84%   |
| 10                        | 84%       | 84%   |
| 15                        | 84%       | 84%   |

### Credit Card Fraud dataset

| Number of boosting rounds | Training  | Test  |
| :---:                     | :---:     | :---: |
| 5                         | 78%       | 78%   |
| 10                        | 98%       | 98%   |
| 15                        | 97%       | 97%   |

