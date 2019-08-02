---
title: "Anomaly Detection"
date: 2019-08-02 13:15:00 +0900
categories: Machine Learning
---


### Credit Fraud Detection

## 0. Import Libraries

## 1. Load dataset
Fistly load the dataset,
```
dataset = pd.read_csv("path/file.csv")
```

Then check the imbalance of the target variable out,
```
sns.countplot(dataset['targetName'])
plt.title("count plot of target variable", fontsize=12)
```
If there is imbalance, it can affect to overfitting the model.      
Even though the model always choose the same choice, we can't know that.

So, we need a technique, 'sampling' to make the distribution of the target, 1:1


```
class1 = dataset.loc[dataset['Class'] == 1]
size = len(class1)
class2 = dataset.loc[dataset['Class'] ==0][:size]
concatDataset = pd.concat([class1, class2])
newDataset = concatDataset.sample(frac=1, random_state=10) # shuffle
```


## 2. Visualize the features
Also, other feaures need to be cheked using simple method.
```
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
amountValue = df['Amount'].values
timeValue = df['Time'].values

ax[0].set_title("Transaction Amount")
sns.distplot(amountValue, ax=ax[0])

ax[1].set_title("Transaction Time")
sns.distplot(timeValue, ax=ax[1])

plt.show()
```

These are the correlation matrices between feature1 and feature2 for all the feature
```
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(20,20))

corr = dataset.corr()
ax[0].set_title('Imbalanced Correlation Matrix')
sns.heatmap(corr, ax=ax[0])

subSampleCorr = newDataset.corr()
ax[1].set_title('Sub-Sample Correlation Matrix')
sns.heatmap(subSampleCorr, ax=ax[1])

plt.show()
```


## 3. Pre-processing
