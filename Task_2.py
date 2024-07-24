import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Generate synthetic dataset
np.random.seed(42)
n_samples = 891

data = {
    'PassengerId': range(1, n_samples + 1),
    'Survived': np.random.choice([0, 1], size=n_samples),
    'Pclass': np.random.choice([1, 2, 3], size=n_samples),
    'Name': [f'Name{i}' for i in range(1, n_samples + 1)],
    'Sex': np.random.choice(['male', 'female'], size=n_samples),
    'Age': np.random.randint(1, 80, size=n_samples),
    'SibSp': np.random.randint(0, 5, size=n_samples),
    'Parch': np.random.randint(0, 5, size=n_samples),
    'Ticket': [f'Ticket{i}' for i in range(1, n_samples + 1)],
    'Fare': np.round(np.random.uniform(10, 500, size=n_samples), 2),
    'Cabin': [f'Cabin{i}' for i in range(1, n_samples + 1)],
    'Embarked': np.random.choice(['C', 'Q', 'S'], size=n_samples)
}

df = pd.DataFrame(data)

# Introduce some missing values
df.loc[np.random.choice(df.index, size=100, replace=False), 'Age'] = np.nan
df.loc[np.random.choice(df.index, size=50, replace=False), 'Cabin'] = np.nan
df.loc[np.random.choice(df.index, size=30, replace=False), 'Embarked'] = np.nan

# Data Cleaning
print("Data Cleaning")

# Handling missing values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Cabin'] = df['Cabin'].fillna('Unknown')
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Correct data types
df['Survived'] = df['Survived'].astype('int')
df['Pclass'] = df['Pclass'].astype('int')

# Removing duplicates
df.drop_duplicates(inplace=True)

# EDA
print("\nExploratory Data Analysis")

# Descriptive statistics
print(df.describe())

# Survival count
sns.countplot(data=df, x='Survived')
plt.title('Survival Count')
plt.show()

# Survival rate by gender
sns.countplot(data=df, x='Survived', hue='Sex')
plt.title('Survival Rate by Gender')
plt.show()

# Age distribution
sns.histplot(df['Age'], bins=30, kde=True)
plt.title('Age Distribution')
plt.show()

# Fare distribution
sns.histplot(df['Fare'], bins=30, kde=True)
plt.title('Fare Distribution')
plt.show()

# Survival rate by passenger class
sns.countplot(data=df, x='Survived', hue='Pclass')
plt.title('Survival Rate by Passenger Class')
plt.show()

# Correlation heatmap
numeric_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Scatter plot of Age vs Fare
sns.scatterplot(data=df, x='Age', y='Fare', hue='Survived')
plt.title('Age vs Fare')
plt.show()

print("Data cleaning and EDA complete.")
