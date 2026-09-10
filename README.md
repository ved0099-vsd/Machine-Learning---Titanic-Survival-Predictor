# Machine-Learning---Titanic-Survival-Predictor
Machine Learning project that predicts Titanic passenger survival using Logistic Regression with data preprocessing, dataset splitting, model training, and model preservation.

# Titanic Survival Predictor

A Machine Learning project that predicts whether a Titanic passenger survived or not using Logistic Regression.

## Project Overview

This project follows a complete Machine Learning workflow, starting from loading the dataset and preprocessing the data to training a Logistic Regression model and preserving the trained model.

## Project Workflow

```text
Load Dataset
     ↓
Data Preprocessing
     ↓
Split Dataset
     ↓
Train Logistic Regression Model
     ↓
Preserve Trained Model
     ↓
Make Predictions
Project Components
1. Load Data

Loads the Titanic dataset and prepares it for further processing.

2. Data Preprocessing

Cleans and prepares the dataset for Machine Learning by handling the required data preprocessing steps.

3. Split Dataset

Splits the dataset into training and testing data for model development and evaluation.

4. Train Model

Trains a Logistic Regression model using the training dataset.

5. Preserve Model

Saves the trained Machine Learning model so that it can be reused for future predictions without retraining.

Technologies Used
Python
Pandas
NumPy
Scikit-learn
Logistic Regression
Pickle
Project Structure
Titanic-Survival-Predictor/
│
├── TitanicLogistic1_LoadData.py
├── TitanicLogistic2_DataPreprocessing.py
├── TitanicLogistic3_SplitDataset.py
├── TitanicLogistic4_TrainModel.py
├── TitanicLogistic5_PreserveModel.py
├── dataset/
└── README.md
How to Run

Run the scripts in the following order:

python TitanicLogistic1_LoadData.py
python TitanicLogistic2_DataPreprocessing.py
python TitanicLogistic3_SplitDataset.py
python TitanicLogistic4_TrainModel.py
python TitanicLogistic5_PreserveModel.py
Machine Learning Algorithm

Logistic Regression is used to predict the survival outcome of Titanic passengers.

The model performs binary classification:

0 → Did Not Survive
1 → Survived
Project Type

Machine Learning / Classification Project

Author

Ved Dhamal


### One important thing

Since your **5 files form one pipeline**, keep all five in the same `Titanic-Survival-Predictor` folder. Don't upload only `TitanicLogistic5_PreserveModel.py`—the README correctly shows the complete workflow.
