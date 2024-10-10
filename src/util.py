import os
import csv

def read_folder_and_write_csv(s2_tiff_folder, is2_csv_file, output_csv):
    # Read the atl file to get the mapped file names
    with open(is2_csv_file, 'r') as f:
        mapped_files = set(line.strip() for line in f)

    # Get all file names in the s2_tiff folder
    file_names = os.listdir(s2_tiff_folder)

    # Write to the CSV file
    with open(output_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['id', 'file_name', 'is_mapped'])

        for idx, file_name in enumerate(file_names, start=1):
            is_mapped = file_name in mapped_files
            csvwriter.writerow([idx, file_name, is_mapped])

# Define paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

print (current_dir, parent_dir)
s2_tiff_folder = os.path.join(parent_dir, 'S2_tiff')
is2_csv_file = os.path.join(parent_dir, 'ATL03_S2_2019-11-01_2019-11-30_ross.csv')
output_csv = os.path.join(parent_dir, 'output.csv')

# # Execute the function
read_folder_and_write_csv(s2_tiff_folder, is2_csv_file, output_csv)
