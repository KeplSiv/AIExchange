# import fitz
# from datetime import datetime

# # Function to convert month abbreviation to month number


# def month_to_number(month_abbr):
#     date = datetime.strptime(month_abbr, '%b')
#     return date.strftime('%m')

# # Function to convert month year string to YYYY-MM format


# def convert_to_yyyymm(month_year_str):
#     date = datetime.strptime(month_year_str, '%b %Y')
#     return date.strftime('%Y-%m')


# # Path to your PDF file
# pdf_file = 'rate.pdf'

# # Open the PDF file
# pdf_document = fitz.open(pdf_file)

# data_formatted = []

# # Replace with actual left position for the number column
# number_left = 154.26565551757812
# # Replace with actual right position for the number column
# number_right = 542.250244140625

# for page_num in range(len(pdf_document)):
#     page = pdf_document.load_page(page_num)
#     text_blocks = page.get_text("blocks")

#     for block in text_blocks:
#         bbox = block[:4]  # Bounding box coordinates
#         text = block[4]  # Text content

#         # Split the text into lines
#         lines = text.split('\n')

#         for line in lines:
#             # Check if the line contains a month abbreviation
#             if any(month in line for month in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']):
#                 parts = line.split()
#                 if len(parts) >= 2:
#                     month_year_str = parts[0] + ' ' + parts[-1]

#                     # Extract number part based on position
#                     number_text = ""
#                     words = line.split()
#                     for word in words:
#                         word_bbox = page.search_for(word)
#                         if word_bbox and len(word_bbox) > 0:
#                             word_bbox = word_bbox[0]
#                             word_left, word_top, word_right, word_bottom = word_bbox

#                             if number_left <= word_left <= number_right:
#                                 number_text += word

#                     number_text = number_text.strip()

#                     try:
#                         # Convert month year string to YYYY-MM format
#                         formatted_date = convert_to_yyyymm(month_year_str)
#                         data_formatted.append(
#                             f"{formatted_date},{number_text}")
#                     except ValueError as e:
#                         print(f"Error parsing date '{month_year_str}': {e}")

# # Close the PDF document
# pdf_document.close()

# # Sort dates from oldest to newest
# data_formatted.sort()

# # dates in YYYY-MM format with corresponding numbers, sorted from oldest to newest
# for date in data_formatted:
#     print(date)

# from datetime import datetime
# import fitz
# --------------------


# import fitz
# from datetime import datetime

# # Function to convert month year string to YYYY-MM format


# def convert_to_yyyymm(month_year_str):
#     try:
#         date = datetime.strptime(month_year_str, '%b %Y')
#         return date.strftime('%Y-%m')
#     except ValueError as e:
#         print(f"Error parsing date '{month_year_str}': {e}")
#         return None


# # Path to your PDF file
# pdf_file = 'rate.pdf'

# # Open the PDF file
# pdf_document = fitz.open(pdf_file)

# data_formatted = []

# # Define the approximate positions for the number column
# number_left = 154.26565551757812
# number_right = 500

# for page_num in range(len(pdf_document)):
#     page = pdf_document.load_page(page_num)
#     text_blocks = page.get_text("blocks")

#     for block in text_blocks:
#         bbox = block[:4]  # Bounding box coordinates
#         text = block[4]  # Text content

#         # Split the text into lines
#         lines = text.split('\n')

#         for line in lines:
#             # Check if the line contains a month abbreviation
#             if any(month in line for month in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']):
#                 parts = line.split()
#                 if len(parts) >= 2:
#                     month_year_str = parts[0] + ' ' + parts[-1]

#                     # Extract number part based on position
#                     number_text = ""

#                     # Search for all words in the line
#                     words = line.split()
#                     for word in words:
#                         # Get the bounding box for the word
#                         word_bbox = page.search_for(word)
#                         if word_bbox and len(word_bbox) > 0:
#                             word_bbox = word_bbox[0]
#                             word_left, _, word_right, _ = word_bbox

#                             # Check if the word's x-position is within the number column range
#                             if number_left <= word_left <= number_right:
#                                 number_text += word + " "

#                                 # Debug output for each word in the number column
#                                 print(
#                                     f"Found '{word}' at position ({word_left}, {word_right})")

#                     number_text = number_text.strip()

#                     # Convert month year string to YYYY-MM format
#                     formatted_date = convert_to_yyyymm(month_year_str)
#                     if formatted_date:
#                         data_formatted.append(
#                             f"{formatted_date},{number_text}")

# # Close the PDF document
# pdf_document.close()

# # Sort dates from oldest to newest
# data_formatted.sort()

# # Print the sorted data_formatted
# for date in data_formatted:
#     print(date)


# Function to convert month year string to YYYY-MM format

import fitz
from datetime import datetime


def convert_to_yyyymm(month_year_str):
    try:
        date = datetime.strptime(month_year_str, '%b %Y')
        return date.strftime('%Y-%m')
    except ValueError as e:
        print(f"Error parsing date '{month_year_str}': {e}")
        return None


# Path to your PDF file
pdf_file = 'rate.pdf'

# Open the PDF file
pdf_document = fitz.open(pdf_file)

data_formatted = []

# Define the approximate positions for the number column
number_left = 550
number_right = 10000
number_top = 25  # Adjust as per your requirement
number_bottom = 10000  # Adjust as per your requirement

for page_num in range(len(pdf_document)):
    page = pdf_document.load_page(page_num)
    text_blocks = page.get_text("blocks")

    for block in text_blocks:
        bbox = block[:4]  # Bounding box coordinates
        text = block[4]  # Text content

        # Split the text into lines
        lines = text.split('\n')

        for line in lines:
            # Check if the line contains a month abbreviation
            if any(month in line for month in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']):
                parts = line.split()
                if len(parts) >= 2:
                    month_year_str = parts[0] + ' ' + parts[-1]

                    # Extract number part based on position
                    number_text = ""

                    # Search for all words in the line
                    words = line.split()
                    for word in words:
                        # Get the bounding box for the word
                        word_bbox = page.search_for(word)
                        if word_bbox and len(word_bbox) > 0:
                            word_bbox = word_bbox[0]
                            word_left, word_top, word_right, word_bottom = word_bbox

                            # Check if the word's x and y positions are within the number column range
                            if (number_left <= word_left <= number_right) and (number_top <= word_top <= number_bottom):
                                number_text += word + " "

                                # Debug output for each word in the number column
                                print(
                                    f"Found '{word}' at position ({word_left}, {word_top})")

                    number_text = number_text.strip()

                    # Convert month year string to YYYY-MM format
                    formatted_date = convert_to_yyyymm(month_year_str)
                    if formatted_date:
                        data_formatted.append(
                            f"{formatted_date},{number_text}")

# Close the PDF document
pdf_document.close()

# Sort dates from oldest to newest
data_formatted.sort()

# Print the sorted data_formatted
for date in data_formatted:
    print(date)
