import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
from imblearn.over_sampling import SMOTE

def check_data_integrity(df):
    print("\n Checking for missing values after processing:")
    print(df.isnull().sum().sort_values(ascending=False).head(20))
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype == 'object':
                df[col] = df[col].fillna('Unknown')
            elif df[col].dtype == 'float64' or df[col].dtype == 'int64':
                df[col] =df[col].fillna(df[col].median())
                
        #else:
            #print(f"Column {col} has no missing values.")
    
   # print("\n Checking for missing values after processing:")
   # print(df.isnull().sum().sort_values(ascending=False).head(20))


    print("\nChecking for number of unique values in each column:")

    for col in df.columns:
        unique_count = df[col].nunique()
        # Drop columns with more than 500 unique values
        if unique_count > 500:
           df.drop(columns=[col], inplace=True)

        #print(f"{col}: {unique_count} unique values")
    
    print("\nChecking the data types of each column:")
    #for col in df.columns:
    #   print(f"{col}: {df[col].dtype}")
    #print(df.columns.to_list())

    # encoding categorical variables
    
    categorical_columns = df.select_dtypes(include=['object']).columns
    print(f"\nCategorical columns before encoding: {categorical_columns}")
    if len(categorical_columns) > 0:
        print(f"One-hot encoding these columns: {categorical_columns}")
        df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    #print(df.columns.to_list())
    #print("\nOne-hot encoding applied to categorical columns.")
    
    print("\nFinal DataFrame info:")
    print(df.info())
    
    return df

    



def train_test_split_data(df):

    #Split the data into features and target variable

    # X is all the columns except 'Severity'
    #print(df.columns.to_list())
    

    X = df.drop(columns=['Severity'])
    #print(X.dtypes)
    #print("\nFeatures (X) columns:", X.columns.to_list())

    # y is the 'Severity' column
    y = df['Severity']

    #non_numeric_columns = X.select_dtypes(include=['object']).columns
    #if len(non_numeric_columns) > 0:
    #    print(f"Non-numeric columns found: {non_numeric_columns}")
    #    raise ValueError("All columns in X must be numeric before training the model.")

    """
    Training with SMOTE (Synthetic Minority Over-sampling Technique) to handle class imbalance
    Should help with the imbalance in the dataset with the level 4 severity being underrepresented
    Dont use SMOTE in the testing phase, only in the training phase
    """
    


    print("Testing is starting")
    start_time = time.time()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    sm = SMOTE(random_state=42)
    X_train_smote, y_train_smote = sm.fit_resample(X_train, y_train)

    # Initialize the Random Forest Classifier
    #n_estimators = number of trees 
    # random_state = ensures reproducibility
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Fit the model on the training data
    model.fit(X_train_smote, y_train_smote)

    print("Before SMOTE:", y_train.value_counts())
    print("After SMOTE:", pd.Series(y_train_smote).value_counts())


    # Make predictions on the test data
    y_pred = model.predict(X_test)
    end_time = time.time()

    print(f"Model training and prediction completed in {end_time - start_time:.2f} seconds.")
    print("Accuracy of the model:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    


