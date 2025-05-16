# python-Analyzing-Data-with-Pandas-and-Visualizing-Results-with-Matplotlib
# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris

# Optional: Settings for plots
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# Task 1: Load and Explore the Dataset
try:
    # Load dataset from sklearn and convert to DataFrame
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)

    print("✅ Dataset loaded successfully.\n")
except Exception as e:
    print(f"❌ Failed to load dataset: {e}")

   # Display first few rows
print("📌 First five rows of the dataset:")
print(df.head()) 

# Data structure and missing values
print("\n📌 Data types and info:")
print(df.info())

print("\n📌 Missing values in the dataset:")
print(df.isnull().sum())

# Clean dataset (No missing values in Iris dataset)
# If there were missing values:
# df = df.dropna() or df.fillna(value)

# Task 2: Basic Data Analysis
print("\n📊 Basic statistics:")
print(df.describe())

# Group by species and compute the mean
grouped_means = df.groupby("species").mean()
print("\n📊 Mean of numerical features grouped by species:")
print(grouped_means)


# Observations
print("\n🔍 Observation:")
print("Setosa generally has smaller features, while Virginica tends to have the largest.")

# Task 3: Data Visualisation

# Line Chart - Simulating trend by using index as time (not ideal for Iris, but for assignment)
plt.plot(df.index, df["sepal length (cm)"], label='Sepal Length')
plt.title("Line Chart: Sepal Length Trend (Index as Time)")
plt.xlabel("Index")
plt.ylabel("Sepal Length (cm)")
plt.legend()
plt.grid(True)
plt.show()

# Bar Chart - Average petal length per species
sns.barplot(x="species", y="petal length (cm)", data=df)
plt.title("Bar Chart: Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.show()

# Histogram - Distribution of sepal width
plt.hist(df["sepal width (cm)"], bins=10, color='skyblue', edgecolor='black')
plt.title("Histogram: Sepal Width Distribution")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.show()

# Scatter Plot - Sepal length vs Petal length
sns.scatterplot(data=df, x="sepal length (cm)", y="petal length (cm)", hue="species")
plt.title("Scatter Plot: Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend()
plt.show()
