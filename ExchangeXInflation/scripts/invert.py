import csv

# Define the input and output file names
input_file = 'exchange_rate.csv'
output_file = 'reversed_data.csv'

# Read the data from the input file
with open(input_file, 'r') as infile:
    reader = csv.reader(infile)
    data = list(reader)

# Reverse the data
reversed_data = data[::-1]

# Write the reversed data to the output file
with open(output_file, 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerows(reversed_data)

print(f"Data has been reversed and written to {output_file}")
