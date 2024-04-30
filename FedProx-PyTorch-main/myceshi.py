import numpy as np
import pandas as pd

df=pd.read_csv("data/Wind/ceshi2.csv",encoding="gbk")
columns = df.columns
print(df.dtypes)
print(df.isnull().sum())
print("==========================")

# 只选择数值列进行填充
numeric_columns = df.select_dtypes(include=['number']).columns

# 检查每列是否存在空值
columns_with_null = df.columns[df[numeric_columns].isnull().any()].tolist()

# 只对存在空值的列进行填充
for column in columns_with_null:
    df.fillna({column:df[column].mean()}, inplace=True)

print(df)
print(df.isnull().sum())