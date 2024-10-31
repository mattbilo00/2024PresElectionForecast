

import os
import requests
import shutil
import pandas as pd
import subprocess


'''
Downloads the 2024 presidential polling from 538 and cleans the dates
'''

_url = "https://projects.fivethirtyeight.com/polls-page/data/president_polls.csv"
_filename = "pres_polls_2024.csv"
_destination = "/Users/mattg/Personal Projects/2024_forecast/Data/"

def fix_dates(date_str):
    try:
        return pd.to_datetime(date_str, format='%m/%d/%y')
    except ValueError:
        return pd.to_datetime(date_str, errors='coerce')

def clean_polling():
    _file_path = _destination + _filename
    _df = pd.read_csv(_file_path)
    columns_to_fix = ['start_date', 'end_date', 'election_date']

    for col in columns_to_fix:
        _df[col] = _df[col].apply(fix_dates)
        _mask = _df[col].dt.year < 2000 # assuming all dates after 2000 are valid
        _df.loc[_mask, col] = _df.loc[_mask, col].apply(lambda x: x.strftime('%m/%d/%y') if pd.notnull(x) else x)
    _df.to_csv(_destination + "pres_polls_2024_clean.csv", index=False)

def download_files():
    response = requests.get(_url)
    if response.status_code == 200:
        with open(_filename, 'wb') as f:
            f.write(response.content)
        print("File downloaded successfully")

        # move file to destination and replace if it exists
        _destination_file_path = os.path.join(_destination, _filename)
        shutil.move(_filename, _destination_file_path)
        print("File moved to:", _destination_file_path)
    else:
        print("File failed to download")

    clean_polling()
    return

download_files()


################################################################
'''
Download the approval numbers of Biden and Harris from 538
'''

# URLs of the CSV files
_urls = [
    "https://projects.fivethirtyeight.com/biden-approval-data/approval_polllist.csv",
    "https://projects.fivethirtyeight.com/biden-approval-data/approval_topline.csv",
    "https://projects.fivethirtyeight.com/polls/data/vp_approval_polls.csv"
]

_new_filenames = ["biden_approval_raw.csv", 
                "biden_approval_topline.csv", 
                "harris_approval_raw.csv"]

# Directory where files will be moved
_destination_dir = "/Users/mattg/Personal Projects/2024_forecast/Data/pres_approval"

# Function to download CSV files and move them
def _download_and_move_csv(url, destination_dir, new_filename):
    response = requests.get(url)
    if response.status_code == 200:
        # Writing to a temporary file in the current directory
        temp_filename = "temp.csv"
        with open(temp_filename, 'wb') as file:
            file.write(response.content)
        # Moving the temporary file to the destination directory with the new filename
        shutil.move(temp_filename, os.path.join(destination_dir, new_filename))
        print(f"Downloaded and moved {new_filename} successfully.")
    else:
        print(f"Failed to download from {url}. Status code: {response.status_code}")

# Download and move CSV files from each URL with the desired filenames
for _url, _new_filename in zip(_urls, _new_filenames):
    _download_and_move_csv(_url, _destination_dir, _new_filename)

# Update timestamp on webpage
subprocess.run(['python', 'update_webpage.py'])

# # Run model v1
# print("Running model v1")
# subprocess.run(['python', 'model_v1_run.py'])

# # Plot model v1
# subprocess.run(['python', 'plot_model_v1.py'])