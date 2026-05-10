# Week 1 Assignment - Axcentra ML Internship
# Titanic Data Cleaning Project
# By: Kanneboyina Premanjali

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Task 1 - Load dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
print("Dataset loaded successfully!")
print(df.info())
print(df.describe())

# Task 2 - Handle missing data
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Fill Age with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill Embarked with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop Cabin (too many missing)
df.drop(columns=['Cabin'], inplace=True)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# Task 3 - Encode categorical variables
le = LabelEncoder()
df['Sex_encoded'] = le.fit_transform(df['Sex'])

# OneHotEncoding for Embarked
df = pd.get_dummies(df, columns=['Embarked'], prefix='Embarked')

print("\nEncoding done!")
print(df[['Sex', 'Sex_encoded']].head())

# Task 4 - Visualize age distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], bins=30, kde=True, color='steelblue')
plt.title('Age Distribution of Titanic Passengers')
plt.xlabel('Age')
plt.ylabel('Count')
plt.savefig('age_distribution.png')
plt.show()
print("Age distribution plot saved!")

# Task 5 - Save cleaned dataset
df.to_csv('titanic_cleaned.csv', index=False)
print("\nCleaned dataset saved as titanic_cleaned.csv")
print(f"Final dataset shape: {df.shape}")