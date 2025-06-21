import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import joblib
import datetime

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
    #Replace the level 4 severity with level 3 since 4 is underrepresented in the dataset
    df['Severity'] = df['Severity'].replace({4:3})
    df['Severity'] = df['Severity'].replace({1:2})  
    # Replace level 1 with level 2 for better balance
    
    df.drop(columns=['Year','End_Hour','Humidity(%)'],inplace = True)  # Ensure no NaN in target variable
    print("\nFinal DataFrame info:")
    print(df.info())
    
    return df

    



def train_test_split_data(df):

    #Split the data into features and target variable
    # X is all columns except 'Severity'
    X = df.drop(columns=['Severity'])
    # y is the 'Severity' column
    y = df['Severity']
    

    """
    Training with SMOTE (Synthetic Minority Over-sampling Technique) to handle class imbalance
    Should help with the imbalance in the dataset with the level 4 severity being underrepresented
    Dont use SMOTE in the testing phase, only in the training phase
    """
    


    print("Testing is starting")
    start_time = time.time()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    #rus = RandomUnderSampler(sampling_strategy={2: 400000, 3: 300000, 1: 50000}, random_state=42)
    #X_train_under,y_train_under = rus.fit_resample(X_train, y_train)
    
    sm = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = sm.fit_resample(X_train, y_train)
 
    # random_state = ensures reproducibility
    # instead of balanced, gonna try class_weight={1: 8, 2:1, 3:4}
    model = RandomForestClassifier(n_estimators=100, random_state=42)  # Using balanced class weights

    
    #model.fit(X_train_under, y_train_under)
    # Fit the model on the training data
    model.fit(X_train_balanced, y_train_balanced)

    print("Before resampling:", y_train.value_counts())
    #print("After undersampling:", pd.Series(y_train_under).value_counts())
    print("After SMOTE:", pd.Series(y_train_balanced).value_counts())


    # Make predictions on the test data
    y_pred = model.predict(X_test)
    end_time = time.time()

    print(f"Model training and prediction completed in {end_time - start_time:.2f} seconds.")
    print("Accuracy of the model:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    feature_importances.nlargest(15).plot(kind='barh')
    plt.title('Top 15 Feature Importances')
    plt.show()
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    joblib.dump(model, f'accident_severity_predictor_{timestamp}.pkl')

    


