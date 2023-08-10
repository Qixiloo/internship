import pandas as pd

# Read the Excel tables
df = pd.read_excel('final_all_time_1.xlsx')


df['违规年度更新'].fillna(df['违规年度'], inplace=True)
df.dropna(subset=['违规年度更新', '违规年度'], how='all', inplace=True)
df.reset_index(drop=True, inplace=True)
df = df[df['违规年度更新'] != "无"]




df['证券代码'] = df['证券代码'].astype(str)
df['违规年度更新']=df['违规年度更新'].astype(str)
df['公告日期']=df['公告日期'].astype(str)

expanded_rows = []
for index, row in df.iterrows():
    # Split the items by ";"
    items = row['违规年度更新'].split('，')
    for item in items:
        new_row = row.copy()  # Create a copy of the original row
        new_row['违规年度更新'] = item.strip()  # Replace the column value with the item
        expanded_rows.append(new_row)  # Append the new row to the list
expanded_df = pd.DataFrame(expanded_rows)
expanded_df.reset_index(drop=True, inplace=True)


expanded_df['证券代码'] = expanded_df['证券代码'].astype(str).str.zfill(6)
expanded_df['公告日期']=expanded_df['公告日期'].astype(str).str[:4]
expanded_df['time_difference'] = expanded_df['公告日期'].astype(int) - expanded_df['违规年度更新'].astype(int)

expanded_df['time_difference'] = expanded_df['time_difference'].astype(str)

# Print the updated DataFrame
expanded_df.to_excel('final_all_time.xlsx', index=False)

