import mysql.connector
import pandas as pd
mydb = mysql.connector.connect(
  host="localhost",
  database='Intern',
  user="root",
  password="YQXGXZ123m",
  auth_plugin='mysql_native_password'
)

cursor = mydb.cursor()


# Read the data from the SQL table into a Pandas DataFrame
df = pd.read_sql_query(f"SELECT * FROM {'db_factor'}", mydb)

# Replace 'your_output_file.csv' with the desired name for the CSV file
output_csv_file = 'db_factor.csv'

# Export the DataFrame to a CSV file
df.to_csv(output_csv_file, index=False)

# Close the connection
mydb.commit()
mydb.close()






