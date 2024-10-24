import os
import csv

# write a script that renames all rows that contain the word "COPERNICUS/S2_SR/" to "" in a new csv file
def rename_rows(file_path, output_file_path):
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
        for row in rows:
            row[:] = [cell.replace("COPERNICUS/S2_SR/", "") for cell in row]
    with open(output_file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

file_path = "ATL03_S2_2019-11-01_2019-11-30_ross.csv"
output_file_path = "ATL03_S2_2019-11-01_2019-11-30_ross_renamed.csv"
# rename_rows(file_path, output_file_path)


def csv_to_dict(files):
    file_map = {}
    for file in files:
        if file in file_map:
            file_map[file] += 1
        else:
            file_map[file] = 1
    return file_map

# write a script that reads the folder S2_tiff and returns a list of all the file names (excluding the first 10 characters) in a python map
def get_file_names(folder_path, file_path):
    files = os.listdir(folder_path)
    files = [file[10:] for file in files]
    # put the files in a map and count frequency if it is a duplicate
    file_map = {}
    for file in files:
        if file in file_map:
            file_map[file] += 1
        else:
            file_map[file] = 1
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
        for row in rows:
            # print row 3
            if row[3] in file_map:
                file_map[row[3]] += 1
            else:
                file_map[row[3]] = 1

t
    return file_map

folder_path = "S2_tiff"
get_file_names(folder_path, output_file_path)