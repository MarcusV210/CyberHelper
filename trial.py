import pandas as pd #type: ignore

df = pd.read_csv('data/CyberSecurity_data.csv')
df.drop(['Unnamed: 0'], axis=1, inplace=True)

