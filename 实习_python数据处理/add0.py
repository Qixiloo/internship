import pandas as pd



# Read tables A, B, and C into separate DataFrames
df_A = pd.read_excel('final_1_time.xlsx')
df_B = pd.read_excel('./final_2_time.xlsx')
df_C = pd.read_excel('./final_all_time.xlsx')

# Merge tables A and B, prioritizing values from table C
combined_df = pd.concat([df_A, df_B, df_C]).drop_duplicates().combine_first(df_C)

# Reset the index of the combined DataFrame
combined_df.reset_index(drop=True, inplace=True)
combined_df = combined_df.drop_duplicates()

# Save the combined DataFrame to a new table
combined_df.to_csv('combined_table.csv', index=False)
combined_df.to_excel('combined_table.xlsx', index=False)
