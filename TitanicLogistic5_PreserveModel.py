import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 1 : Load Data

#-----------------------------------------------------
#   Function Name : LoadData
#   Description :   Load the data from CSV
#   Input :         Name of csv file
#   Output :        Data frame
#   Author :        Vedant Dhamal
#   Date :          16/08/2026
#-----------------------------------------------------
def LoadData(filename):
    df = pd.read_csv(filename)

    print("Dataset loaded succesfully")
    print(df.head())

    return df

# Step 2 : Data Preprocessing

#-----------------------------------------------------
#   Function Name : PreprocessData
#   Description :   It performs data analysis
#   Input :         Data Frame
#   Output :        Updated Data frame
#   Author :        Vedant Dhamal
#   Date :          16/08/2026
#-----------------------------------------------------

def PreprocessData(df):
    df = df.drop(
    columns=[
        "Passengerid",
        "zero",
        "name"
    ],
    errors="ignore"
)

    # Handle missing values
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    #Convert categorical to numeric data
    df = pd.get_dummies(
        df,
        columns=["Embarked"],
        drop_first= True,
        dtype=int
    )

    print(df.head())

    print("Data prepsocessing completed")

    return df

# Step 3 : Split Data 

#-----------------------------------------------------
#   Function Name : SplitData
#   Description :   It performs Splitting activity
#   Input :         Data Frame
#   Output :        4 subsets for traiing and testing
#   Author :        Vedant Dhamal
#   Date :          16/08/2026
#-----------------------------------------------------

def SplitData(df):
    X = df.drop("Survived", axis = 1)
    Y = df["Survived"]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42
    )

    print("Dataset Splitting completed succesfully")

    return X_train, X_test, Y_train, Y_test


# Step 4 : Train the model

#-----------------------------------------------------
#   Function Name : TrainModel
#   Description :   It performs model training
#   Input :         Training fetures and labels
#   Output :        Trained model
#   Author :        Vedant Dhamal
#   Date :          16/08/2026
#-----------------------------------------------------

def TrainModel(X_train, Y_train):
    model = LogisticRegression(max_iter=1000)

    model = model.fit(X_train, Y_train)

    print("Model trained succesfully")

    return model


# Step 5 : Evaluate model

#-----------------------------------------------------
#   Function Name : EvaluateModel
#   Description :   It performs model testing
#   Input :         model, testing data (fetures ,labels)
#   Output :        none
#   Author :        Vedant Dhamal
#   Date :          16/08/2026
#-----------------------------------------------------

def EvaluateModel(model, X_test, Y_test):
    Y_pred = model.predict(X_test)

    accuracy = accuracy_score(Y_test,Y_pred)

    print("Accuracy is : ",accuracy)

    print(confusion_matrix(Y_test,Y_pred))

# Step 6 : Preserve Model

#-----------------------------------------------------
#   Function Name : PreserveModel
#   Description :   It performs model preservation into .pkl file
#   Input :         model
#   Output :        none
#   Author :        Vedant Dhamal
#   Date :          16/08/2026
#-----------------------------------------------------

def PreserveModel(model,filename):
    joblib.dump(model,filename)

    print("Model preserved with name : ",filename)

#-----------------------------------------------------
#   Function Name : main
#   Description :   Entry point function
#   Input :         None
#   Output :        None
#   Author :        Vedant Dhamal
#   Date :          16/08/2026
#-----------------------------------------------------
def main():
    # Step 1 
    df = LoadData("MarvellousTitanicDataset.csv")

    # Step 2
    df = PreprocessData(df)

    # Step 3
    X_train, X_test, Y_train, Y_test = SplitData(df)

    # Step 4 : 
    model = TrainModel(X_train, Y_train)

    # Step 5 : 
    EvaluateModel(model,X_test,Y_test)

    # Step 6 : 
    PreserveModel(model,"MarvellousTitanic.pkl")

if __name__ == "__main__":
    main()