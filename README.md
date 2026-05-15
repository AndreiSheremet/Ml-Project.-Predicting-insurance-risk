# Insurance Risk Prediction

This project focuses on predicting insurance risk using structured claims data.  
The goal was to clean and explore the dataset, build several machine learning models, and compare their performance.

## Project Overview

The project includes data cleaning, exploratory data analysis, model training, model evaluation, and a final written report.

Three machine learning approaches were tested:

- Decision Tree
- Feedforward Neural Network (FFNN)
- XGBoost

The models were compared based on their predictive performance and interpretability.

## Repository Structure

```text
.
├── DataCleaning and Exploration.ipynb
├── Decision_Tree_model.ipynb
├── FFNN_model.ipynb
├── XgBoost_model.ipynb
├── model_functions/
├── claims_train.csv
├── claims_test.csv
├── Insurance_ML_Project.pdf
└── README.md

**Files and Folders**
DataCleaning and Exploration.ipynb - Contains the data cleaning and exploratory data analysis (EDA).

Decision_Tree_model.ipynb - Notebook used to train, test, and evaluate the Decision Tree model.

FFNN_model.ipynb - Notebook used to train, test, and evaluate the Feedforward Neural Network model.

XgBoost_model.ipynb - Notebook used to train, test, and evaluate the XGBoost model.

model_functions/ - Contains my own supporting code for the Decision Tree and FFNN models. This folder includes reusable functions used during model training, testing, and evaluation.

claims_train.csv and claims_test.csv - Datasets used for training and testing the models.

Insurance_ML_Project.pdf - Final report describing the project, methodology, experiments, model results, and conclusions.
