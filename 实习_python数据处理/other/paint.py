import pandas as pd
import jieba
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

table=pd.read_excel('./3e.xlsx')
table['违规年度']=table['违规年度'].replace('N/A;',np.nan).fillna(0).astype(int)

print(table['公告日期'].dtype)
print(table['uyear'].dtype)
print(table['违规年度'].dtype)

table['year_difference'] = table.apply(
    lambda row: (row['uyear'] - row['公告日期']) if row['uyear'] != 0
    else (row['违规年度']- row['公告日期']),
    axis=1
)


code_counts = table['year_difference'].value_counts().reset_index()
code_counts.columns = ['year_difference', 'count']

# Display the code counts
code_counts.to_csv('count_year.csv', index=False)

t_table=pd.read_excel('./3e.xlsx')

t_table['违规类型'] = t_table['违规类型'].str.split('、|;')
code_counts = t_table['违规类型'].explode().value_counts().reset_index()
code_counts.columns = ['违规类型', 'count']

# Display the code counts
code_counts.to_csv('count_type.csv', index=False)

