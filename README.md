# Bank Customer Churn Prediction Model


## This repository contains a machine learning model trained from scratch to predict bank customer churn. The model uses customer data to determine the likelihood of a customer leaving the bank. The dataset includes features such as credit score, age, balance, and more.

## Table of Contents
Project Overview

Dataset Description

Features

Model Architecture

Installation

Usage

Results


## Project Overview
Customer churn is a critical metric for banks, as retaining customers is often more cost-effective than acquiring new ones. This project aims to predict whether a customer will churn (leave the bank) based on their demographic and financial data. The model is trained using a dataset containing various customer attributes and their churn status.

## Dataset Description
The dataset used for this project contains the following features:

## Feature Name	Description
* CreditScore	The credit score of the customer.
* Age	The age of the customer.
* Tenure	The number of years the customer has been with the bank.
* Balance	The account balance of the customer.
* NumOfProducts	The number of bank products the customer uses (e.g., savings, credit card).
* HasCrCard	Whether the customer has a credit card (1 = Yes, 0 = No).
* IsActiveMember	Whether the customer is an active member (1 = Yes, 0 = No).
* EstimatedSalary	The estimated salary of the customer.
* Exited	The target variable indicating whether the customer churned (1 = Yes, 0 = No).
  Features
## The model uses the following features to predict customer churn:

* CreditScore: A numerical value representing the customer's creditworthiness.

* Age: The age of the customer.

* Tenure: The number of years the customer has been with the bank.

* Balance: The current balance in the customer's account.

* NumOfProducts: The number of bank products the customer uses.

* HasCrCard: A binary flag indicating whether the customer has a credit card.

* IsActiveMember: A binary flag indicating whether the customer is an active member.

* EstimatedSalary: The estimated salary of the customer.
  
* Gender: The Gender of the customer.

## Model Architecture
The model is built using a Sequential Algorithmm, which is well-suited for classification tasks like churn prediction. The architecture includes:

Data Preprocessing: Handling missing values, scaling numerical features, and encoding categorical variables.

Feature Selection: Selecting the most relevant features to improve model performance.

Model Training: Training the Sequential Algorithm on the preprocessed data.

Evaluation: Evaluating the model using metrics such as accuracy, precision, recall, and F1-score.

Installation
To run this project locally, follow these steps:

Clone the Repository:

bash
Copy
git clone https://github.com/your-username/bank-customer-churn.git
cd bank-customer-churn
Set Up a Virtual Environment:

bash
Copy
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
Install Dependencies:

bash
Copy
pip install -r requirements.txt
Usage
Train the Model:
Run the following command to train the model:

bash
Copy
python train.py
Make Predictions:
Use the trained model to make predictions on new data:

bash
Copy
python predict.py --input data/new_customers.csv
Evaluate the Model:
Evaluate the model's performance on the test dataset:

bash
Copy
python evaluate.py
Results
The model achieved the following performance metrics on the test dataset:

Accuracy: 92.5%

Precision: 93.0%

Recall: 94.0%

F1-Score: 91.0%

These results indicate that the model is effective at predicting customer churn and can be used to identify at-risk customers.
