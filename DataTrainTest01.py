# pip install pandas

import pandas as pd
import os
import zipfile

def prepare_csv_from_zip(data_dir, csv_filename, zip_filename):
    csv_path = os.path.join(data_dir, csv_filename)
    zip_path = os.path.join(data_dir, zip_filename)
    if not os.path.exists(csv_path):
        print(f"'{csv_path}' is not exist. Checking Zip file.")
        if os.path.exists(zip_path):
            print(f"'{zip_path}' is found. Unzip...")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(data_dir)
                print(f"Completed Unzip. '{csv_path}' will be used.")
            except Exception as e:
                print(f"Unzip error: {e}")
                exit()
        else:
            print(f"Error: '{csv_filename}' and '{zip_filename}'are not found.")
            exit()
    return csv_path

data_dir = './preprocessed'
data_csv_path = prepare_csv_from_zip(data_dir, 'KoreaApartDeal_PreProcessed.csv', 'KoreaApartDeal_PreProcessed.zip')

try:
    df = pd.read_csv(data_csv_path)
except Exception as e:
    print(f"file read error: {e}")
    exit()


# todo : train ...