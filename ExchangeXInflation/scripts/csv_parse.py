import pandas as pd

# Read the CSV file
df = pd.read_csv('usd_inr.csv', header=None, names=['date', 'number'])

# Convert the 'date' column to datetime with the correct format
df['date'] = pd.to_datetime(df['date'], format='%d/%m/%y')

# Format the date to 'yyyy-mm'
df['date'] = df['date'].dt.strftime('%Y-%m')

# Save the updated DataFrame back to CSV
df.to_csv('exchange_rate.csv', index=False)
