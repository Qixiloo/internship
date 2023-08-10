import pandas as pd

# Read the Excel tables
marked_table = pd.read_excel('mark.xlsx')

m_null_count = marked_table['myear'].notnull().sum()
#print(m_null_count)
updated_table = pd.read_excel('update.xlsx')

# Create a dictionary mapping IDs to marked information
marked_info_dict = dict(zip(marked_table['mID'], marked_table['upyear']))


# Update the need_to_be_updated_information column in the updated table
for index, row in updated_table.iterrows():
    id_value = row['uID']
    if id_value in marked_info_dict:
        marked_info = marked_info_dict[id_value]
        updated_table.at[index, 'uyear'] = marked_info

# Save the updated table to a new Excel file
updated_table.to_excel('updated_table_updated.xlsx', index=False)


updated = pd.read_excel('updated_table_updated.xlsx')
non_null_count = updated['uyear'].notnull().sum()

# Print the count
print(non_null_count)






