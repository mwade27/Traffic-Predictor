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
import os

def binning_columns(df):
    # Need to bin the 'Temperature(F)', 'Humidity(%)', and 'Visibility(mi)', Pessure, Precipitation and windSpeedcolumns
    temp_bins = [-np.inf, 10, 20, 30, 40, 50, 60, 70, 80, 90, np.inf]
    temp_labels = [
    'Extreme Cold', 'Frigid', 'Freezing', 'Cold', 'Chilly',
    'Mild', 'Warm', 'Hot', 'Very Hot', 'Extreme Heat']
    df['Temperature_Binned'] = pd.cut(df['Temperature(F)'], bins=temp_bins, labels=temp_labels)
    
    humid_bins = [-np.inf,30,50,85,100, np.inf]
    humid_labels = ['Dry', 'Comfortable', 'Humid', 'Very Humid', 'Oppressive']
    df['Humidity_Binned'] = pd.cut(df['Humidity(%)'], bins=humid_bins, labels=humid_labels)
    
    press_bins = [-np.inf, 29.5, 29.9, 30.2, 30.5, np.inf]
    press_labels =['Low', 'Below Normal', 'Normal', 'High', 'Very High']
    df['Pressure_Binned'] = pd.cut(df['Pressure(in)'], bins=press_bins, labels=press_labels)
    
    vis_bins = [-np.inf, 0.5, 1, 2, 4, 6, 10, 15, np.inf]
    vis_labels = [
    'Zero', 'Very Poor', 'Poor', 'Low', 'Moderate',
    'High', 'Very High', 'Excellent']
    df['Visibility_Binned'] = pd.cut(df['Visibility(mi)'], bins=vis_bins, labels=vis_labels)
    
    windspeed_bins = [-np.inf, 5, 10, 15, 20, 30, np.inf]
    windspeed_labels = ['Very Low', 'Low', 'Moderate', 'High', 'Very High', 'Extreme']
    df['Wind_Speed_Binned'] = pd.cut(df['Wind_Speed(mph)'], bins=windspeed_bins, labels=windspeed_labels)
    
    precip_bins = [-np.inf, 0, 0.01, 0.1, 0.25, 0.5, 1, 2, 5, np.inf]
    precip_labels = ['None', 'Trace', 'Light', 'Moderate', 'Heavy','Very Heavy', 'Extreme', 'Severe', 'Catastrophic']
    df['Precipitation_Binned'] = pd.cut(df['Precipitation(in)'], bins=precip_bins, labels=precip_labels)
    
    windchill_bins = [-np.inf, 0, 10, 20, 32, 40, 50, 60, 70, np.inf]
    windchill_labels = ['Extreme Cold', 'Frigid', 'Freezing', 'Cold', 'Chilly','Mild', 'Warm', 'Hot', 'Very Hot']
    df['Wind_Chill_Binned'] = pd.cut(df['Wind_Chill(F)'], bins=windchill_bins, labels=windchill_labels)

    hour_bins = [0,7,10, 16, 19, 24]
    hour_labels = ['Early_Morning','Morning_Rush', 'Midday', 'Evening_Rush', 'Night']
    df['Start_Hour_Binned'] = pd.cut(df['Start_Hour'], bins=hour_bins, labels=hour_labels)
    #add binning for Precipitation and Wind_Chill
    df.drop(columns=['Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)','Precipitation(in)','Wind_Chill(F)','Start_Hour'], inplace=True, errors='ignore')
    
    print("\n Binning Completed:")
    return df

def check_data_integrity(df):
    print("\n Checking for missing values after processing:")
    df.dropna(subset=['Start_Hour_Binned'], inplace=True)
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
    #exclude_columns = ['Severity','Temperature_Binned', 'Humidity_Binned', 'Pressure_Binned', 'Visibility_Binned', 'Wind_Speed_Binned']
   
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
    
    categorical_columns = df.select_dtypes(include=['object','category']).columns
    print(f"\nCategorical columns before encoding: {categorical_columns}")
    if len(categorical_columns) > 0:
        print(f"One-hot encoding these columns: {categorical_columns}")
        df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    #print(df.columns.to_list())
    #print("\nOne-hot encoding applied to categorical columns.")
    
    df['Severity'] = df['Severity'].replace({4:3,1:2})
    
    
    df.drop(columns=['Year','End_Hour','Start_Lat','Start_Lng','End_Date','End_Hour','Start_Date'],axis=1,inplace = True,errors='ignore')  # Ensure no NaN in target variable
    #print("\nFinal DataFrame info:")
    #print(df.columns.to_list())
    
    return df

    


#Thursday might be good to test with different estimators, 100,125,150 or test out XGBoost
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
   
    sm = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = sm.fit_resample(X_train, y_train)
 
    # random_state = ensures reproducibility
    model = RandomForestClassifier(n_estimators=100, random_state=42,class_weight = 'balanced')  # Using balanced class weights
    
   
   
    # Fit the model on the training data
    model.fit(X_train_balanced, y_train_balanced)
   
    # 📏 Tree Depth Analysis
    depths = [estimator.tree_.max_depth for estimator in model.estimators_]
    print(f"\nTree Depths → Avg: {np.mean(depths):.2f}, Max: {np.max(depths)}")
   
    print("Before resampling:", y_train.value_counts())
    #print("After undersampling:", pd.Series(y_train_under).value_counts())
    print("After SMOTE:", pd.Series(y_train_balanced).value_counts())
    y_train_pred = model.predict(X_train_balanced)
    train_accuracy = accuracy_score(y_train_balanced, y_train_pred)
    train_error = 1 - train_accuracy
   
    # Make predictions on the test data
    
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_error = 1 - train_accuracy
   


   
    print("Accuracy of the model:", test_accuracy)
    print("\nClassification Report:", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    feature_importances.nlargest(15).plot(kind='barh')
    plt.title('Top 15 Feature Importances')
    plt.show()
    
    end_time = time.time()
    print(f"Model training and prediction completed in {end_time - start_time:.2f} seconds.")
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    joblib.dump(model, f'accident_severity_predictor_{timestamp}.pkl')

    
    
    
    
    




