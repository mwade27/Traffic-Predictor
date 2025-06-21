import pandas as pd
from eda import severity_distribution, weather_condition_distribution, top_accident_locations, accident_by_timeofday
from mlmodel import check_data_integrity, train_test_split_data
#May need to drop Start_time and End_time later after feature engineering
filepath = 'data/US_Accidents_March23.csv'
def load_data(filepath):
    df = pd.read_csv(filepath)
    print("Data loaded successfully.")
    
    columns_to_drop = ['Source','Description','Civil_Twilight','Nautical_Twilight','Country',
                       'Weather_Timestamp','Wind_Direction','Zipcode', 'Airport_Code','Astronomical_Twilight','ID']
    # Drop unnecessary columns
    df.drop(columns = columns_to_drop, axis=1, inplace=True)
    #print(df.head())
    return df

""" 
Handle missing values 
"""
def handle_missing_values(df):
    # Check for missing values
    print("\n Top 20 columns with missing Values ")
    print(df.isnull().sum().sort_values(ascending=False).head(20))
    
    # Drop rows with missing values in 'Severity' column since its are target variable
    df.dropna(subset=['Severity'], inplace=True)
    # Calculate the percentage of missing values in each column
    missing = df.isnull().mean()
    high_missing_cols = missing[missing > 0.4].index
    df.drop(columns=high_missing_cols, inplace=True)
    
    # Fill missing values for categorical and numerical columns
    cat_columns = df.select_dtypes(include=['object']).columns
    for col in cat_columns:
        df[col] = df[col].fillna("Unknown")
        
    num_columns = df.select_dtypes(include=['float64', 'int64']).columns
    for col in num_columns:
        df[col] = df[col].fillna(df[col].median())
        
#Splitting time columns into separate features for better analysis and visualization
def time_splitting(df):
    # Convert 'Start_Time' and 'End_Time' to datetime format
    # starttime splitting
    df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors='coerce') 
    df['Start_Date'] = df['Start_Time'].dt.date
    df['Start_Hour'] = df['Start_Time'].dt.hour
    df['Month'] = df['Start_Time'].dt.month_name()
    df['Year'] = df['Start_Time'].dt.year
    # May add Day of the week later
    
    # endtime splitting 
    df["End_Time"] = pd.to_datetime(df["End_Time"], errors='coerce')
    df['End_Date'] = df['End_Time'].dt.date
    df['End_Hour'] = df['End_Time'].dt.hour
    # Dont need to add month and day of the week
    
   
#Convert boolean columns to integers (0 and 1) for better compatibility with Power BI and other tools
def convert_booleans(df):
    bool_columns = df.select_dtypes(include=['bool']).columns
    df[bool_columns] = df[bool_columns].astype(int)
    print(f"Converted boolean columns to integers.{list(bool_columns)}")
    
# if locations are highly correlated, drop one of them    
def dropredundantlocations(df):
    if 'Start_Lat' in df.columns and 'End_Lat' in df.columns:
        correlationlocations = df[['Start_Lat', 'End_Lat']].corr().iloc[0,1]
        if correlationlocations > 0.95:
            df.drop(columns=['End_Lat' , 'End_Lng'], inplace=True)
            print("Dropped 'End_Lat' due to high correlation with 'Start_Lat'.")
        else:
            print("No high correlation found between 'Start_Lat' and 'End_Lat'.")
    else:
        print("Columns 'Start_Lat' or 'End_Lat' not found in the DataFrame.")

def time_features(df):
    df['Day_of_Week'] = df['Start_Time'].dt.day_name()
    df['Is_Weekend'] = df['Day_of_Week'].isin(['Saturday', 'Sunday']).astype(int)
     #drop the original 'Start_Time' and 'End_Time' columns
    df.drop(columns=['Start_Time', 'End_Time'], axis=1 ,inplace=True)
    
        
def feature_engineering(df):
    convert_booleans(df)
    dropredundantlocations(df)
    time_features(df)



    # drop Start_Lat and Start_Lng,Distance,City and Street since they have over 10k unique values
    


    
def main():
    # Load the dataset
    df = load_data('data/US_Accidents_March23.csv')
    handle_missing_values(df)
    time_splitting(df)
    feature_engineering(df)
    
    #print(df.columns)
    
    #print(df_ml.columns)
    
    #df_ml = df.copy()
    #df_ml = check_data_integrity(df_ml)
    #print(df['Severity'].value_counts())
    #train_test_split_data(df_ml)
    
    
    #print("\n Data Info after processing")
    #print(df_ml.info())
    #print(df_ml.columns.to_list())
  
    
   
    
    
    #severity_distribution(df)
    #weather_condition_distribution(df)
    #top_accident_locations(df)
    #accident_by_timeofday(df)
    
    print(df.columns.to_list())
    
    #uncomment later but for this will run faster
    df.to_csv('data/cleaned_US_Accidents_March23.csv', index=False)
    

    #print("\n Data Info")
    #print(df.info())
    
    #print("Summary Statistics")
    #print(df.describe(include='all'))
    
    
if __name__ == "__main__":
    main()