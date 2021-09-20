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
The task is to train a loan prediction classifier using the dataset. Customer churn or customer attrition is the phenomenon where customers of a business no longer purchase or interact with the business. 

The ability to be able to predict that a certain customer is at a very high risk of churning, while there is still some time to do something significant about it, itself represents a great additional potential revenue source for any business.

So, in this task we will be predicting whether a customer will churn or not and if yes, then the concerned team can inquire or further analyse on cutomer behavior and purchase details.

### Access

The dataset was downloaded from this [Github Repo](https://github.com/fcgomezr/Udacity-Project-3/blob/main/data%20sets/WA_Fn-UseC_-Telco-Customer-Churn.csv) where I have staged it for direct download to the AML workspace using SDK.

Once the dataset was downloaded, SDK was again used to clean and split the data into training and validation datasets, which were then stored as Pandas dataframes in memory to facilitate quick data exploration and query, and registered as AML [TabularDatasets](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.data.dataset_factory.tabulardatasetfactory?view=azure-ml-py) in the workspace to enable remote access by the AutoML experiment running on a remote compute cluster.

The dataset after downloaded and registered into the workspace looks like this:

![](https://github.com/fcgomezr/Udacity-Project-3/blob/main/Screen%20Shops/Datasets.png)

## Automated ML

Automated machine learning picks an algorithm and hyperparameters for you and generates a model ready for deployment. There are several options that you can use to configure automated machine learning experiments.
This is a binary classification problem with label column 'Attrition' having output as 'true' or 'false'. 25 mins is the experiment_timeout_duration, a maximum of 5 concurrent iterations take place together, and the primary metric is AUC_weighted.

The AutoML configurations used for this experiment are:

| Auto ML Configuration | Value | Explanation |
|    :---:     |     :---:      |     :---:     |
| experiment_timeout_minutes | 25 | Maximum duration in minutes that all iterations can take before the experiment is terminated |
| max_concurrent_iterations | 5 | Represents the maximum number of iterations that would be executed in parallel |
| primary_metric | AUC_weighted | This is the metric that the AutoML runs will optimize for when selecting the best performing model |
| compute_target | cpu_cluster(created) | The compute target on which we will run the experiment |
| task | classification | This is the nature of our machine learning task |
| training_data | dataset(imported) | The training data to be used within the experiment |
| label_column_name | Attrition | The name of the label column |
| path | ./capstone-project | This is the path to the project folder |
| enable_early_stopping | True | Enable early termination if the score is not improving |
| featurization | auto | Indicates if featurization step should be done automatically or not, or whether customized featurization should be used |
| debug_log | automl_errors.log | The log file to write debug information to |

The AutoML experiment run was executed with this AutoMLConfig settings:



```python
# TODO: Put your automl settings here
automl_settings = {    
    "experiment_timeout_minutes": 25,
    "max_concurrent_iterations": 5,
    "primary_metric" : 'AUC_weighted'}

# TODO: Put your automl config here
automl_config = AutoMLConfig(
                             compute_target=comp_trget,
                             task = "classification",
                             training_data=temp,
                             label_column_name="Churn",   
                             enable_early_stopping= True,
                             featurization= 'auto',
                             n_cross_validations=7,
                             debug_log = "automl_errors.log",
                             **automl_settings
)
```

The experiment ran on a remote compute cluster with the progress tracked real time using the `RunDetails` widget as shown below:

![](https://github.com/fcgomezr/Udacity-Project-3/blob/main/Screen%20Shops/AML%20-%20Best%20Model.png)

The experiment run took slightly over **27 minutes** to finish:

![](https://github.com/fcgomezr/Udacity-Project-3/blob/main/Screen%20Shops/AML%20-%20Time.png)
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
