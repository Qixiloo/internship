import pandas as pd

# Read the four tables into separate DataFrames
table1 = pd.read_excel('data/Debit_FI_T1.xlsx')
table2 = pd.read_excel('data/Develop_FI_T8.xlsx')
table3 = pd.read_excel('data/Manage_FI_T4.xlsx')
table4 = pd.read_excel('data/profit_FI_T5.xlsx')

table1['Stkcd'] = table1['Stkcd'].astype(str).str.zfill(6)
table1=table1[table1['Typrep']=='A']
table1 = table1[table1['Accper'].str.endswith('-12-31')]


table2['Stkcd'] = table2['Stkcd'].astype(str).str.zfill(6)
table2=table2[table2['Typrep']=='A']
table2 = table2[table2['Accper'].str.endswith('-12-31')]


table3['Stkcd'] = table3['Stkcd'].astype(str).str.zfill(6)
table3=table3[table3['Typrep']=='A']
table3 = table3[table3['Accper'].str.endswith('-12-31')]

table4['Stkcd'] = table4['Stkcd'].astype(str).str.zfill(6)
table4=table4[table4['Typrep']=='A']
table4 = table4[table4['Accper'].str.endswith('-12-31')]

# Merge the tables based on 'StockId' and 'Year' columnsx


table1.set_index(['Stkcd', 'Accper'], inplace=True)

# Set the 'Stkcd' column as the index in the other tables
table2.set_index(['Stkcd', 'Accper'], inplace=True)
table3.set_index(['Stkcd', 'Accper'], inplace=True)
table4.set_index(['Stkcd', 'Accper'], inplace=True)

merged_df = table1.join(table2, how='outer', rsuffix='_2')
merged_df = merged_df.join(table3, how='outer', rsuffix='_3')
merged_df = merged_df.join(table4, how='outer', rsuffix='_4')



# Reset the index to turn the primary keys into regular columns
merged_df.reset_index(inplace=True)

# Save the merged table to a new CSV file
merged_df.to_csv('merged_table.csv', index=False)
merged_df.to_excel('merged_table.xlsx', index=False)
