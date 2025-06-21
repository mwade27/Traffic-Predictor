import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def severity_distribution(df):
    sns.countplot(x='Severity', data=df)
    plt.title('Distribution of Accident Severity')
    plt.xlabel('Severity Level')
    plt.ylabel('Number of Accidents')
    plt.show()



def weather_condition_distribution(df):
    sns.countplot(data=df, x='Weather_Condition', order=df['Weather_Condition'].value_counts().head(10).index, hue='Severity')
    plt.title('Severity of Accidents based on Weather Conditions')
    plt.show()
    
    top_conditions = df['Weather_Condition'].value_counts().head(10).index
    filtered_df = df[df['Weather_Condition'].isin(top_conditions)]
    sns.barplot(data=filtered_df, x='Weather_Condition', y='Severity', order=top_conditions)
    plt.title('Average Severity by Weather Condition')
    plt.xlabel('Weather Condition')
    plt.ylabel('Average Severity')
    plt.xticks(rotation=45)
    plt.show()

def top_accident_locations(df):
    top_10cities = df['City'].value_counts().head(20)
    top_10cities.plot(kind='bar')
    plt.title('Top 10 Accident Cities')
    plt.xlabel('City')
    plt.ylabel('Number of Accidents')
    plt.show()
    
    top_10states = df['State'].value_counts().head(20)
    top_10states.plot(kind='bar')
    plt.title('Top 10 Accident States')
    plt.xlabel('State')
    plt.ylabel('Number of Accidents')
    plt.show()
    
def accident_by_timeofday(df):
    sns.histplot(df['Start_Hour'], bins=24, kde=True)
    plt.title('Accidents by Time of Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Accidents')
    plt.show()


