# Predicting Diabetes based on different health indicators

## Overview

This project contains a Jupyter Notebook for EDA and 2 .py files for classification logic. Below is a step-by-step guide on how to set up, run, and use these files. No need of additional arguments. The csv file is included in the folder and code can read it.

### Files Overview:

1. **`avinash_nelluri_EDA.ipynb`**: Exploratory Data Analysis of the dataset. 
2. **`avinash_nelluri_all_classifiers.py`**: Contains logic to run different regression and classifiers, but without handling class imbalance and feature selection
3. **`avinash_nelluri_handle_imbalance_feature_selection.py`**: Contains logic to run different regression and classifiers, with handling class imbalance and feature selection

---

## Prerequisites

Make sure you have the following installed before running the code:
```
pandas, seabond, matlpotlib, pyspark, xgboost
```
---

## How to Run the Code

### 1. **avinash_nelluri_EDA.py**: Exploratory Data Analysis of the dataset


**Command**:
```bash
jupyter notebook avinash_nelluri_EDA.ipynb
```
### 2. **avinash_nelluri_all_classifiers.py**: ML models without handling class imbalance

**Command**:
```bash
python avinash_nelluri_all_classifiers.py
```

### 3. **avinash_nelluri_handle_imbalance_feature_selection.py**: ML models with handling class imbalance and feature selection


**Command**:
```bash
python avinash_nelluri_handle_imbalance_feature_selection.py
```