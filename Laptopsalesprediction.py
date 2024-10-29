import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
df = pd.read_csv(r"C:\Users\Lenovo\OneDrive\Desktop\Banu Docs\laptop_cleaned2.csv")
df
df.drop("Unnamed: 0",axis=1,inplace=True)
df.isnull().sum()
df.shape
df["Name"][555]
df.drop(["Graphics_GB","Total_processor","Execution_units","Processor_gen"],axis=1,inplace=True)
df.shape
obj_col = df.select_dtypes(include="object").columns
df.isnull().sum()
null_col = df.columns[df.isnull().any()].tolist()
from sklearn.impute import SimpleImputer
col = list(df.columns)
df.dtypes
null_col
si = SimpleImputer(strategy="most_frequent")
float_null = list(df[null_col].select_dtypes(exclude="object"))
df_si = si.fit_transform(df[null_col])
df_si.shape
df_si=pd.DataFrame(df_si,columns=null_col)
for i in float_null:
  df_si[i]=df_si[i].astype('float64')
  df_si.dtypes
for i in null_col:
  df[i] = df_si[i]
  df.isnull().sum()
  df.dtypes
  np.random.seed(42)
df["Sales_vol"] = (1000 / (df["Price"] + 1)) * (1 + df["Rating"] / 5)+np.random.randint(0,2)
sns.histplot(df["Sales_vol"])
Y = df["Sales_vol"]
X = df.drop("Sales_vol",axis=1)
ohe = OneHotEncoder(sparse_output=False)
list(X.select_dtypes(include="float").columns)
col = list(X.select_dtypes(include="object").columns)
float_columns = list(X.select_dtypes(include="float").columns)
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(sparse_output=False), col),
        ('num', 'passthrough', float_columns)
    ]
)
transformed_data = preprocessor.fit_transform(df)
list(X.select_dtypes(exclude="object").columns)
df.dtypes
transformed_data
x_train,x_test,y_train,y_test = train_test_split(transformed_data,Y,test_size=0.2,random_state=42)
model = XGBRegressor(n_estimators=1000, early_stopping_rounds=10)
model.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=False)
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("RMSE:", rmse)
np.sqrt(mean_squared_error(y_train,model.predict(x_train)))
company = []
for i in range(df.shape[0]):
    company.append(df["Name"][i].split()[0])
company = pd.DataFrame(company)
df['Brand'] = company[0]
plt.figure(figsize=(10, 6))
plt.bar(df['Brand'], df['Sales_vol'])
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
plt.figure(figsize=(10, 6))
plt.bar(df['Brand'], df['Price'])
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
plt.figure(figsize=(10, 6))
plt.ylim(3.5, 5)
plt.bar(df['Brand'], df['Rating'])
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
