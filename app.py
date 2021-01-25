import pandas as pd

df = pd.read_csv('car data.csv')
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', len(df.columns))
print(df.head())
