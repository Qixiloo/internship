import pandas as pd
import pyodbc

df = pd.read_excel('FINAL_sorted.xlsx')

df['证券代码'] = df['证券代码'].astype(str).str.zfill(6)
df = df.drop_duplicates(subset=['证券代码', '公告日期', '违规年度'])

df.to_excel('1.xlsx', index=False)