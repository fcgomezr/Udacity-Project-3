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


The best model generated by AutoML experiment run was the `VotingEnsemble` model:

![](https://github.com/fcgomezr/Udacity-Project-3/blob/main/Screen%20Shops/AML%20-%20Results%201.png)


The best performing model is the `VotingEnsemble` with an AUC_weighted value of **85,84%**. A voting ensemble (or a “majority voting ensemble“) is an ensemble machine learning model that combines the predictions from multiple other models. It is a technique that may be used to improve model performance, ideally achieving better performance than any single model used in the ensemble. This balances out the individual weaknesses of the considered classifiers.

![](https://github.com/fcgomezr/Udacity-Project-3/blob/main/Screen%20Shops/AML%20-%20Results%202.png)

### Improve AutoML Results
* Increase experiment timeout duration: This would allow for more model experimentation, but might be costly.
* Try a different primary metric: We can explore other metrics like `f1 score`, `log loss`, `precision - recall`, etc. depending on the nature of the data you are working with.
* Engineer new features that may help improve model performance..
* Explore other AutoML configurations.

## Hyperparameter Tuning

 The machine learning model I have chosen to go along with hyperdrive is [Sckit-learn LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)  model. In `LogisticRegression`, these were the two hyperparameters tuned by `HyperDrive` during the experiment run. 

The `max_current_runs` was set to 5 and `mex_total_runs` was set to 40 to ensure the experiment can fit in the chosen experiment timeout limit of 30 minutes.


The HyperDrive experiment run was configured with [parameter](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriveconfig?view=azure-ml-py#parameters) settings as follows:
  
  
</p>  
  Some of the hyperdrive configuration done for this classification task are:
  <ul>
    <li> The parameter sampler I chose is <b>LogisticRegression</b> because it supports both discrete and continuous hyperparameters. It supports early termination of low-          performance runs and supports early stopping policies. </li>

  </ul>
</p>   



```python
 # Specify parameter sampler
   ps = RandomParameterSampling(
    {
    '--C' : choice(0.03,0.3,3,10,30),
    '--max_iter' : choice(25,50,75,100)
   })
  ```
  
  
  
</p>
</ul>
    <li> The early stopping policy I chose was <b>BanditPolicy</b> because it is based on slack factor and evaluation interval. Bandit terminates runs where the primary metric            is not within the specified slack factor compared to the best performing run. </li>
</p>
</ul> 
    
```python
# Specify a Policy
policy = BanditPolicy(evaluation_interval = 3, slack_factor = 0.1)
```   
</p>
</ul>
    <li> max_concurrent_runs (4): The maximum number of runs to execute concurrently. </li>
    <li> max_total_runs (40): The maximum total number of runs to create. This is the upper bound; there may be fewer runs when the sample space is smaller than this value.         </li> Finally, <b>HyperDriveConfig</b> using the estimator
  </ul>
</p>

```python
# Create a HyperDriveConfig using the estimator, hyperparameter sampler, and policy.
hyperdrive_config = HyperDriveConfig (
    hyperparameter_sampling=ps,
    primary_metric_name='Accuracy',
    primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
    max_total_runs=40,
    max_concurrent_runs=5,
    policy=policy,
    estimator=est
)
```   


The python training script `train.py` was executed during the experiment run. It downloaded the dataset from this [Github Repo](https://github.com/fcgomezr/Udacity-Project-3/blob/main/data%20sets/WA_Fn-UseC_-Telco-Customer-Churn.csv) , split it into train and test sets, accepted two input parameters `C` and `max_iter` (representing Regularization Strength and Max iterations respectively) for use with [Sckit-learn LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).


### Benefits of the parameter sampler chosen
The [random parameter](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.randomparametersampling?view=azure-ml-py) sampler for HyperDrive supports discrete and continuous hyperparameters, as well as early termination of low-performance runs. It is simple to use, eliminates bias and increases the accuracy of the model.

### Benefits of the early stopping policy chosen
The early termination policy [BanditPolicy](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.banditpolicy?view=azure-ml-py&preserve-view=true#&preserve-view=truedefinition) for HyperDrive automatically terminates poorly performing runs and improves computational efficiency. It is based on slack factor/slack amount and evaluation interval and cancels runs where the primary metric is not within the specified slack factor/slack amount compared to the best performing run.

The experiment ran on a remote compute cluster with the progress tracked real time using the `RunDetails` widget as shown below:


![](https://github.com/fcgomezr/Udacity-Project-3/blob/main/Screen%20Shops/HIPER-%20Rundtails.png)
![](https://github.com/fcgomezr/Udacity-Project-3/blob/main/Screen%20Shops/HIPER-%20Rundtails2.png)


### Results
The best model generated by HyperDrive experiment run was `Run 18` with an accuracy of **82.03%**. The metrics and visulaization charts are as shown below:

![](https://github.com/fcgomezr/Udacity-Project-3/blob/main/Screen%20Shops/HIPER-%20Results%201.png)
![](https://github.com/fcgomezr/Udacity-Project-3/blob/main/Screen%20Shops/HIPER-%20Results%202.png)
![](https://github.com/fcgomezr/Udacity-Project-3/blob/main/Screen%20Shops/HIPER-%20Results%203.png)
![](https://github.com/fcgomezr/Udacity-Project-3/blob/main/Screen%20Shops/HIPER-%20Results%204.png)


## Improve HyperDrive Results
- Choose a different algorithm to train the dataset on like `Random Forest` , `xgboost`, etc.
- Choose a different clssification metric to optimize for
- Choose a different termination policy like `No Termination`, `Median stopping`, etc.
- Specify a different sampling method like `Bayesian`, `Grid`, etc.



## Model Deployment
From above two ways (`automl` and `LogisticRegression`), we see that the best ml model was Voting Ensemble model generated by automl. I have registered and deployed this model as a web service using ACI (Azure Container Instance).The below snapshot shows active web service endpoint that is use to access the deployed machine learning model. The sample data feeded to the deployed model as a web service request as shown in the below screen recording is transformed and encoded data.


To deploy the model, go to the automl notebook and execute the code cells below the markdown like this: 

```python
from azureml.core.model import Model
model = Model.register(workspace = ws, model_name = 'best_fit_automl_model', model_path = 'best_fit_automl_model.pkl')
print(model.name, model.id, model.version, sep='\t')
```

>* Send a request to the web service you deployed to test it. 

```python
from azureml.core.environment import Environment
from azureml.core.model import InferenceConfig
from azureml.core.webservice import LocalWebservice, Webservice, AciWebservice
from azureml.core.conda_dependencies import CondaDependencies
import azureml.train.automl

# Create the environment
env = best_automl.get_environment()
conda_dep = CondaDependencies()


# Adds dependencies to PythonSection of myenv
#env.python.conda_dependencies=conda_dep

inference_config = InferenceConfig(entry_script='score.py', environment=env)

deployment_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=4, enable_app_insights=True)
service = Model.deploy(ws, "customerservice", [model], inference_config, deployment_config)
service.wait_for_deployment(show_output = True)

print(service.state)
print(service.scoring_uri)
print(service.swagger_uri)
```

>* After executing the above block of code cells, the model was deployed as a web service and appeared on the Endpoints dashboard like the screenshots shown below:
>![](https://github.com/fcgomezr/Udacity-Project-3/blob/main/Screen%20Shops/model%20deploy%204.png)
>![](https://github.com/fcgomezr/Udacity-Project-3/blob/main/Screen%20Shops/model%20deploy%205.1.png)
>![](https://github.com/fcgomezr/Udacity-Project-3/blob/main/Screen%20Shops/model%20deploy%205.2.png)

>* Test the scoring endpoint 

```python
data = {"data":
        [{
                "gender_Male" :0,
                "SeniorCitizen":1,
                "Dependents" :0,
                "Partner" :0,
                "tenure":2,
                "PhoneService" :1,
                "MultipleLines" :1,
                "InternetService_Fiber optic" :1,
                "InternetService_No":0,
                "OnlineSecurity" :0,
                "OnlineBackup" :1,
                "DeviceProtection" :0,
                "TechSupport" :0,
                "StreamingTV" :0,
                "StreamingMovies" :1,
                "Contract_One year" : 0,
                "Contract_Two year":0,
                "PaperlessBilling" :1,
                "PaymentMethod_Bank transfer (automatic)":0,
                "PaymentMethod_Credit card (automatic)":0,
                "PaymentMethod_Mailed check":0,
                "PaymentMethod_Electronic check" :1,
                "MonthlyCharges" : 55.7,
                "TotalCharges" : 239.8 }
        ]
        }
```

>* Make the request and display the response

```python
resp = requests.post(scoring_uri, input_data, headers=headers)

print("Response Code : ", resp.status_code)
print("Predicted Value : ",resp.text)
```
>* Make the function define in `score.py`
>```python
>def run(data):
>    try:
>        temp = json.loads(data)
>        data = pd.DataFrame(temp['data'])
>        result = deploy_model.predict(data)
>        # You can return any data type, as long as it is JSON serializable.
>        return result.tolist()
>    except Exception as e:
>        error = str(e)
>        return error
>```


The resulst are this 
![https://github.com/fcgomezr/Udacity-Project-3/blob/main/Screen%20Shops/model%20deploy%203.png]
## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
