# ==========================
# Import Libraries
# ==========================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid", font_scale=1.1)
# ==========================
# Load Cleaned Dataset
# ==========================

df = pd.read_csv(r"D:\nti project\NTI_Graduation_Project\data\cleaned_data.csv")
print(df.head())
print(df.shape)

# ==========================
# Exploratory Data Analysis (EDA)
# ==========================

# 1. Distribution of House Prices
plt.figure(figsize=(10,6))

sns.histplot(df['price'],
             bins=30,
             kde=True,
             color='royalblue')

plt.title("Distribution of House Prices",fontsize=15,fontweight='bold')
plt.xlabel("Price")
plt.ylabel("Count")

plt.tight_layout()
plt.show()

# 2. Box Plot of House Prices
plt.figure(figsize=(8,6))

sns.boxplot(y=df['price'],
            color='skyblue')

plt.title("Box Plot of House Prices",fontsize=15,fontweight='bold')

plt.tight_layout()
plt.show()

# 3. Correlation Heatmap
plt.figure(figsize=(14,10))

corr=df.corr(numeric_only=True)

sns.heatmap(corr,
            cmap='coolwarm',
            annot=True,
            fmt=".2f")

plt.title("Correlation Heatmap",fontsize=15,fontweight='bold')

plt.tight_layout()
plt.show()

# 4 .Correlation with Price
 
corr=df.corr(numeric_only=True)['price'].sort_values()

plt.figure(figsize=(8,10))

corr.plot(kind='barh')

plt.title("Correlation with Price",fontsize=15,fontweight='bold')

plt.tight_layout()
plt.show()

# 5. Living Area vs Price

plt.figure(figsize=(10,6))

sns.scatterplot(data=df,
                x='sqft_living',
                y='price',
                alpha=0.6)

plt.title("Living Area vs Price",fontsize=15,fontweight='bold')

plt.tight_layout()
plt.show()

# 6. House Price by Grade

plt.figure(figsize=(10,6))

sns.boxplot(data=df,
            x='grade',
            y='price',
            palette='Blues')

plt.title("House Price by Grade",fontsize=15,fontweight='bold')

plt.tight_layout()
plt.show()

# 7. House Price by Waterfront

plt.figure(figsize=(8,6))

sns.boxplot(data=df,
            x='waterfront',
            y='price',
            palette='Set2')

plt.title("House Price by Waterfront",fontsize=15,fontweight='bold')

plt.tight_layout()
plt.show()

# 8. House Price by Condition

plt.figure(figsize=(10,6))

sns.boxplot(data=df,
            x='condition',
            y='price')

plt.title("House Price by Condition",fontsize=15,fontweight='bold')

plt.tight_layout()
plt.show()


# 9. House Price by view

plt.figure(figsize=(10,6))

sns.boxplot(data=df,
            x='view',
            y='price')

plt.title("House Price by View",fontsize=15,fontweight='bold')

plt.tight_layout()
plt.show()

# 10. Average Price by Bedrooms

plt.figure(figsize=(10,6))

sns.barplot(data=df,
            x='bedrooms',
            y='price')

plt.title("Average Price by Bedrooms",fontsize=15,fontweight='bold')

plt.tight_layout()
plt.show()

# 11 .Number of Houses by Bedrooms

plt.figure(figsize=(10,6))

sns.countplot(data=df,
              x='bedrooms')

plt.title("Number of Houses by Bedrooms",fontsize=15,fontweight='bold')

plt.tight_layout()
plt.show()

# 12. House Locations

plt.figure(figsize=(10,8))

sns.scatterplot(
    x='long',
    y='lat',
    hue='price',
    data=df,
    palette='viridis',
    alpha=0.6
)

plt.title("House Locations Colored by Price")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.tight_layout()
plt.show()

# 13. Pair Plot

sns.pairplot(df[['price',
                 'sqft_living',
                 'sqft_lot',
                 'grade',
                 'bathrooms']])

plt.show()


