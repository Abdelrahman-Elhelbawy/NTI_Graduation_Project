import pandas as pd
import numpy as np

df=pd.read_csv('data/kc_house_data.csv')
print("Head of data",df.head())
print("_______________________________________________________________")
print("Tail of data",df.tail())
print("_______________________________________________________________")
print("Shape of data",df.shape)
print("_______________________________________________________________")
print("Info of data",df.info())
print("_______________________________________________________________")
print("Describe of data",df.describe())
print("_______________________________________________________________")
print("Types of data",df.dtypes)
print("_______________________________________________________________")
print("Number of null values",df.isnull().sum())
print("_______________________________________________________________")
print("Number of duplicated",df.duplicated().sum())
print("_______________________________________________________________")

df['date'] = pd.to_datetime(df['date'], format='%Y%m%dT%H%M%S')

# بنستخرج منه سنة وشهر البيع كأعمدة رقمية مفيدة للموديل
df['sale_year'] = df['date'].dt.year
df['sale_month'] = df['date'].dt.month

df['was_renovated'] = (df['yr_renovated'] != 0).astype(int)



columns_to_cap = ['bedrooms', 'bathrooms', 'sqft_lot', 'sqft_lot15']

for col in columns_to_cap:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    n_low = (df[col] < lower_bound).sum()
    n_high = (df[col] > upper_bound).sum()

    df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
    df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

    print(f"[{col}] lower={lower_bound:.2f}, upper={upper_bound:.2f} "
          f"-> capped {n_low} low values & {n_high} high values")



df= df.drop(columns=['id', 'date','yr_renovated'])
print("_______________________________________________________________")
print("Shape after cleaning ", df.shape)
