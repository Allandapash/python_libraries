# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

try:
    # Load Iris dataset from sklearn
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    
    # Display first few rows
    print("First 5 rows of dataset:")
    print(df.head(), "\n")
    
    # Explore structure (data types, missing values)
    print("Dataset Info:")
    print(df.info(), "\n")
    
    print("Missing Values per Column:")
    print(df.isnull().sum(), "\n")
    
    # Handle missing values (if any, here none exist)
    df = df.dropna()  

except FileNotFoundError:
    print("Error: Dataset file not found!")
except Exception as e:
    print("An error occurred while loading the dataset:", e)

# -------------------------
# Task 2: Basic Data Analysis
# -------------------------

# Summary statistics
print("Summary Statistics:")
print(df.describe(), "\n")

# Group by species and compute mean of numerical columns
grouped = df.groupby("species").mean()
print("Mean values per species:")
print(grouped, "\n")

# Interesting finding (example)
print("Observation: Virginica species generally has larger sepal and petal dimensions compared to Setosa.\n")



# -------------------------
# Task 3: Data Visualization
# -------------------------

# Line chart (trend of sepal length as row index increases - simulating time series)
plt.figure(figsize=(8,5))
plt.plot(df.index, df["sepal length (cm)"], label="Sepal Length")
plt.title("Line Chart: Sepal Length Trend")
plt.xlabel("Index (simulated time)")
plt.ylabel("Sepal Length (cm)")
plt.legend()
plt.show()

# Bar chart: Average petal length per species
plt.figure(figsize=(8,5))
sns.barplot(x="species", y="petal length (cm)", data=df, estimator=np.mean)
plt.title("Bar Chart: Average Petal Length by Species")
plt.xlabel("Species")
plt.ylabel("Average Petal Length (cm)")
plt.show()

# Histogram: Distribution of sepal width
plt.figure(figsize=(8,5))
plt.hist(df["sepal width (cm)"], bins=20, color="skyblue", edgecolor="black")
plt.title("Histogram: Sepal Width Distribution")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.show()

# Scatter plot: Sepal length vs Petal length
plt.figure(figsize=(8,5))
sns.scatterplot(x="sepal length (cm)", y="petal length (cm)", hue="species", data=df, palette="Set2")
plt.title("Scatter Plot: Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.show()