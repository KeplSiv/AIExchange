import pandas as pd

excel_file = 'Book1.xlsx'

# Load data from Excel file
df = pd.read_excel(excel_file, header=None)

# Prepare empty lists for CSV output
dates = []
inflation_rates = []

# Iterate through each column in the DataFrame
for col in df.columns:
    # Extract year-month information from the first row
    year_month = df.iloc[0, col]
    if pd.notna(year_month):  # Check if it's not NaN
        year_month_str = str(year_month)

        # Split the string to separate year and month
        year = year_month_str[:4]  # Extract year
        month = year_month_str[4:]  # Extract month

        # Remove any decimal point from month if present
        if '.' in month:
            month = month.split('.')[0]

        # Construct date string (formatting month as two digits)
        # Ensure month is two digits (zero-padded)
        date_str = f'{year}-{month.zfill(2)}'

        # Extract inflation rate from the second row
        inflation_rate = df.iloc[1, col]

        # Debug: Print extracted values for verification
        # print(
        #     f'Column {col}: Date={date_str}, Inflation Rate={inflation_rate}')

        # Append to lists
        dates.append(date_str)
        inflation_rates.append(inflation_rate)

# Create DataFrame for CSV output
csv_data = pd.DataFrame({'Date': dates, 'InflationRate': inflation_rates})

# Save dataframe to CSV file with headers included
csv_file = 'inflation_data.csv'
# Ensure headers are included
csv_data.to_csv(csv_file, index=False, header=True)

print(f'CSV file saved: {csv_file}')
