import pandas as pd

# Read the CSV file
df = pd.read_csv('your_csv_file.csv')  # Replace 'your_csv_file.csv' with the path to your CSV file

# Initialize the product mapping dictionary
product_mapping = {}

# Iterate over the rows of the DataFrame and populate the product mapping dictionary
for index, row in df.iterrows():
    product_id = row['product_id']
    image_url = row['image_url']
    purchase_link = row['purchase_link']
    
    # Add the product mapping entry to the dictionary
    product_mapping[product_id] = {
        'image_url': image_url,
        'purchase_link': purchase_link
    }

print(product_mapping)
