*NOTE:* This file is a template that you can use to create the README for your project. The *TODO* comments below will highlight the information you should be sure to include.

# Udacity Azure ML Nanodegree Capstone Project - Predict Customer Churn

## Overview

This Capstone project is part of the Udacity Azure ML Nanodegree.
In this project, I used a loan Application Prediction dataset from Kaggle [(Dataset)](https://www.kaggle.com/blastchar/telco-customer-churn) to build a loan application prediction classifier.The goal Predict behavior to retain customers. I can analyze all relevant customer data and develop focused customer retention programs.

I built two models, one using AutoML and one, custom model. The AutoML is equipped to train and produce the best model on its own, while the custom model leverages HyperDrive to tune training hyperparameters to deliver the best model. Between the AutoML and Hyperdrive experiment runs, a best performing model was selected for deployment. Scoring requests were then sent to the deployment endpoint to test the deployed model.


The diagram below provides an overview of the workflow:
![workflow](https://www.element61.be/sites/default/files/assets/insights/mlservices/Workflow.jpg)


## Project Set Up and Installation
To set up this project, you will need the following 5 items:
> * an Azure Machine Learning Workspace with Python SDK installed
>
> * the two project notebooks named `automl` and `hyperparameter_tuning`
>
> * the python script file named `train.py`
>
> * the conda environment yaml file `conda_env.yml` and scoring script `score.py`
>
To run the project,
> * upload all the 5 items to a jupyter notebook compute instance in the AML workspace and place them in the _**same folder**_
>
> * open the `automl` notebook and run each code cell _**in turn**_ from Section 1 thru 3, _**stop right before**_ Section 4 `Model Deployment`
>
> * open the `hyperparameter_tuning` and run each code cell _**in turn**_ from Section 1 thru 3, _**stop right before**_ Section 4 `Model Deployment`
>
> * compare the best model accuracy in `automl` and `hyperparameter_tuning` notebooks, run Section 4 `Model Deployment` from the notebook that has the _**best**_ performing model
>


## Dataset

### Overview
<p> Each row represents a customer, each column contains customer’s attributes described on the column Metadata. The data set includes information about:
  <ol>
    <li> Customers who left within the last month – the column is called Churn. </li>
    <li> Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and                                                            movies. </li>
    <li> Customer account information – how long they’ve been a customer, contract, payment method, paperless billing, monthly charges, and total charges. </li>
    <li> Demographic info about customers – gender and if they have partners and dependents. </li>
  </ol>
</p>


### Task
*TODO*: Explain the task you are going to be solving with this dataset and the features you will be using for it.

### Access
*TODO*: Explain how you are accessing the data in your workspace.

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
