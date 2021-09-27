from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

# TODO: Create TabularDataset using TabularDatasetFactory
# Data is located at:
 
dir_web = "https://raw.githubusercontent.com/fcgomezr/Udacity-Project-3/main/data%20sets/WA_Fn-UseC_-Telco-Customer-Churn.csv"
ds = TabularDatasetFactory.from_delimited_files(path=dir_web)



def clean_data(data):
    
    # Clean and one hot encode data
    df = data.to_pandas_dataframe().dropna()
    #df['TotalCharges'] = pd.to_numeric(df.TotalCharges, errors='coerce')

    df['Partner'] = df.Partner.apply(lambda s: 1 if s == True else 0)
    df['Dependents'] = df.Dependents.apply(lambda s: 1 if s == True else 0)
    df['PaperlessBilling'] = df.PaperlessBilling.apply(lambda s: 1 if s == True else 0)
    df['PhoneService'] = df.PhoneService.apply(lambda s: 1 if s == True else 0)
    df['MultipleLines'] = df.MultipleLines.apply(lambda s: 1 if s == "Yes" else 0)
    df['OnlineSecurity'] = df.OnlineSecurity.apply(lambda s: 1 if s == "Yes" else 0)
    df['OnlineBackup'] = df.OnlineBackup.apply(lambda s: 1 if s == "Yes" else 0)
    df['DeviceProtection'] = df.DeviceProtection.apply(lambda s: 1 if s == "Yes" else 0)
    df['TechSupport'] = df.TechSupport.apply(lambda s: 1 if s == "Yes" else 0)
    df['StreamingMovies'] = df.StreamingMovies.apply(lambda s: 1 if s == "Yes" else 0)
    df['StreamingTV'] = df.StreamingTV.apply(lambda s: 1 if s == "Yes" else 0)
    
    df = pd.get_dummies(df,columns=['gender'],drop_first= True)
    df = pd.get_dummies(df,columns=['InternetService'],drop_first= True)
    df = pd.get_dummies(df,columns=['PaymentMethod'],drop_first= False)
    df = pd.get_dummies(df,columns=['Contract'],drop_first= True)
    
    y_df = df.pop("Churn").apply(lambda s: 1 if s == True else 0)
    
    return df,y_df


x, y = clean_data(ds)

# TODO: Split data into train and test sets.

### YOUR CODE HERE ###a
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)
run = Run.get_context()

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))

    os.makedirs('outputs',exist_ok= True)
    joblib.dump(model,'outputs/model.joblib')

if __name__ == '__main__':
    main()