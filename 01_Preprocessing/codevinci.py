import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

def visualize_histograms(df, columns):
    for column in columns:
        plt.figure()
        plt.hist(df[column], bins=20)
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.title(f'Histogram: {column}')
        plt.savefig(f'histogram_{column}.png')
        plt.close()

df = pd.read_csv('diabetes_dataset.csv')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = df[features]

# Split dataset into two different dataframes
dataset = df[df['Outcome'] == 0]
dataset1 = df[df['Outcome'] == 1]

# Calculate medians
median = dataset.median()
median1 = dataset1.median()

# Fill missing values using median -> 63%
dataset = dataset.fillna(median)
dataset1 = dataset1.fillna(median1)
merge_tables= pd.concat([dataset, dataset1])
  
# Define a function for treating outliers using IQR
def treat_outliers_iqr(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (column >= lower_bound) & (column <= upper_bound)

#columns_to_exclude = ['Outcome']
#for column in merge_tables.columns:
    if column not in columns_to_exclude:
        merge_tables = merge_tables[treat_outliers_iqr(merge_tables[column])]

merge_tables.info()

outcome_column = merge_tables['Outcome']

merge_tables_normalized = merge_tables
merge_tables_normalized = (merge_tables_normalized - merge_tables_normalized.min()) / (merge_tables_normalized.max() - merge_tables_normalized.min())

merge_tables_normalized.info()
merge_tables_normalized.to_csv("diabetes_dataset_normalized.csv", index=False)

# Visualize histograms
key = False
columns_to_visualize = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
if key:
    visualize_histograms(merge_tables_normalized, columns_to_visualize)