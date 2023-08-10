import pandas as pd


t_table=pd.read_excel('all.xlsx')
new_table=pd.read_excel('updated_table_updated.xlsx')
new_dict = dict(zip(new_table['uID'], new_table['uyear']))

for index, row in t_table.iterrows():
    id_value = row['ID']
    if id_value in new_dict:
        marked_info = new_dict[id_value]
        t_table.at[index, 'uyear'] = marked_info

t_table.to_excel('new.xlsx', index=False)

updated = pd.read_excel('new.xlsx')
non_null_count = updated['uyear'].notnull().sum()


# Print the count
print(non_null_count)
