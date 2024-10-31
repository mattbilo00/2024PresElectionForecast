## Will run Model V1 - USE THIS - Faster version

## This will take an hour to run and delete all your files, please only run when necessary

import os
import csv
import glob
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from concurrent.futures import ThreadPoolExecutor

import warnings
warnings.filterwarnings("ignore")

# Construct the path to match all .txt files
txt_files = glob.glob(os.path.join("/Users/mattg/Personal Projects/2024_forecast/Data/forecast_results/v1/h2h", "*.txt"))
# Loop through the list of .txt files and delete them
for file in txt_files:
    try:
        os.remove(file)
    except Exception as e:
        print(f"Error deleting {file}: {e}")

# Filter for only the columns in cols
df = pd.read_csv("/Users/mattg/Personal Projects/2024_forecast/Data/pres_polls_2024_clean.csv")
cols = ['poll_id', 'question_id', 'state', 'start_date', 'end_date', 'answer', 'pct', 'sample_size', 'population', 'subpopulation', 'population_full', 'election_date', 'party', 'candidate_id', 'candidate_name', 'url']
df = df[cols]

# Fill missing state values and convert date columns to datetime
df['state'] = df['state'].fillna("National")
df['start_date'] = pd.to_datetime(df['start_date'])
df['end_date'] = pd.to_datetime(df['end_date'])
df['election_date'] = pd.to_datetime(df['election_date'])

# Calculate the mid-date of each poll and filter polls after 2024-07-01
df['mid_date'] = pd.to_datetime((df['start_date'] + (df['end_date'] - df['start_date']) / 2).dt.date)
df = df[df['mid_date'] > pd.to_datetime('2024-07-01')]

# Load state lean data
state_lean_path = '/Users/mattg/Personal Projects/2024_forecast/Data/election_results/leans/state_partisan_leans.csv'
state_lean_df = pd.read_csv(state_lean_path)

'''Sate vs National polling weight functions'''
def f(x):
    return (2 / np.pi) * np.arctan(x)

def g(x):
    return (x**2) / (x**2 + 1)

def h(x):
    return np.tanh(x)

def z(x):
    return 1 - np.exp(-x)

# Weight function to balance state and national polling
def weight(x, a):
    b = (4 - a) / 3
    return (a * h(x) + b * (f(x) + g(x) + z(x))) / 4

'''Functions for incorporating sample size weight in pct averages'''
# Function of how to weight poll by sample size
median_sample_size = df['sample_size'].median()
def find_weight(n):
    return np.sqrt(n) / np.sqrt(median_sample_size)

# Calculate weighted averages of polling percentages
def calc_avgs_by_weight(df):
    df = df.groupby(['mid_date', 'answer']).apply(
    lambda x: x['weighted_pct'].sum() / x['weight'].sum()).reset_index(name='pct')
    df.columns = ['mid_date', 'answer', 'pct']
    df = df.sort_values(by='mid_date', ascending=True)
    return df

# Creates a df that includes the count of the number of polls conducted in a day
def sum_poll_count(df):
    df['poll_count'] = df.groupby(['mid_date', 'answer'])['mid_date'].transform('size')
    df = df.sort_values(by='mid_date', ascending=True)
    return df[['mid_date', 'answer', 'poll_count']].drop_duplicates()

def calc_poll_avgs(df):
    # Perform grouping once
    df_grouped = df.groupby(['mid_date', 'answer']).apply(
        lambda x: x['weighted_pct'].sum() / x['weight'].sum()
    ).reset_index(name='pct')
    
    # Compute poll count directly after the same groupby
    df_poll_count = df_grouped.groupby(['mid_date', 'answer'])['mid_date'].size().reset_index(name='poll_count')

    df_grouped = df_grouped.sort_values(by='mid_date', ascending=True)
    df_avg = df_grouped.merge(df_poll_count[['mid_date', 'poll_count']], on='mid_date', how='left')
    
    return df_avg.drop_duplicates()

# Model V1 for calculating and plotting polling data
def model_v1(df, poll_count_alpha, poll_avg_alpha, a):
    
    # Apply sample size weights to polls
    df['weight'] = np.sqrt(df['sample_size']) / np.sqrt(median_sample_size)
    df['weighted_pct'] = df['pct'] * df['weight']

    state_polls = {}
    states = df.state.unique().tolist()

    # Moves National polls first
    states.remove("National")
    states.insert(0, "National")

    # Placeholder for national poll data
    harris_national = trump_national = full_dates = None
    alpha = 0.77

    for state in states:
        state_lean_df['state'] = state_lean_df['state'].str.lower()

        # Filter by state and H2H between Harris and Trump
        df_state = df[df['state'] == state]
        df_state = df_state[df_state['question_id'].isin(
            df_state['question_id'].value_counts()[lambda x: x == 2].index)]
        df_state = df_state.sort_values(by='mid_date').groupby('question_id').filter(
            lambda x: set(x['answer']) == {'Harris', 'Trump'}
        )

        # Ensure mid_date is in datetime format
        df_state['mid_date'] = pd.to_datetime(df_state['mid_date'], errors='coerce')
        # Set mid_date as the index
        df_state.set_index('mid_date', inplace=True)

        # Gets the full date ranges from National polls
        if state == "National":
            full_dates = pd.date_range(df_state.index.min(), df_state.index.max(), freq='D')
        df_state.reset_index(inplace=True)

        if not df_state.empty:
            df_state_avg = calc_poll_avgs(df_state)
        else:
            ## needs to be done separetly
            continue

        # Compute averages and poll counts
        df_state_avg = df_state.groupby(['mid_date', 'answer']).agg(
        avg_pct=('pct', 'mean'),                # Calculate the mean (average) of pct
        poll_count=('pct', 'count')).unstack()  # Count how many polls are in each group

        # Flatten the column MultiIndex
        df_state_avg.columns = df_state_avg.columns.get_level_values(1)
        df_state_avg.columns = ['Harris', 'Trump', 'poll_count', 'drop_col']
        df_state_avg.drop(columns = ['drop_col'], inplace=True)

        # Reindex df_state_avg to include all dates in the range (this introduces NaNs for missing dates)
        df_state_avg = df_state_avg.reindex(full_dates)

        # Interpolate missing values in 'Harris' and 'Trump' columns
        df_state_avg[['Harris', 'Trump']] = df_state_avg[['Harris', 'Trump']].interpolate(method='linear', limit_direction='both')

        # If necessary, reset the index so 'mid_date' becomes a column again
        df_state_avg.reset_index(inplace=True)
        df_state_avg.rename(columns={'index': 'mid_date'}, inplace=True)

        # Compute EWMAs for polling pcts
        df_state_avg["Harris_ewma_pct"] = df_state_avg['Harris'].ewm(alpha=poll_avg_alpha, adjust=False).mean()
        df_state_avg['Trump_ewma_pct'] = df_state_avg['Trump'].ewm(alpha=poll_avg_alpha, adjust=False).mean()
        
        if state == "National":
            df_state_avg.drop(columns=['poll_count'], inplace=True)
            national_df = df_state_avg.copy()

            # Append df to csv file
            upload_df = national_df[['mid_date', 'Harris_ewma_pct', 'Trump_ewma_pct']].copy()

        else:
            '''
            1. Filter state lean for current state
            2. Use time interpolation on current state polls if possible
            3. Average state polls by day, include count of polls per day
            4. Create EWMA for polls and for count of polls
            5. Join state and national polls on mid date
            6. Use a count weight function to determine the weighting of state and national polls to produce an average
            7. When there is limited data (one poll, last poll was months ago, etc) assume the state poll continues flat
            '''
            ## Create the poll count average
            # Identify the first index of a non-NaN value
            first_non_nan_index = df_state_avg['poll_count'].first_valid_index()
            # If there's a valid index, replace NaNs after that index with 0
            if first_non_nan_index is not None:
                df_state_avg.loc[first_non_nan_index+1:, 'poll_count'] = df_state_avg.loc[first_non_nan_index+1:, 'poll_count'].fillna(0)

            # Creates the ewma for polling counts
            # Vary alpha from 0.35 to 0.45
            df_state_avg['poll_count_ewma'] = df_state_avg['poll_count'].ewm(alpha=poll_count_alpha, adjust=False).mean()

            # Applies the weight to the poll count based on above functions
            df_state_avg['state_poll_weight'] = df_state_avg['poll_count_ewma'].apply(lambda x: weight(x, a))

            # Gets the state lean factors
            state = state.lower()
            state_lean = state_lean_df[state_lean_df['state'] == state]
            dem_lean_fac = state_lean['state_lean_factor_dem'].iloc[0]
            gop_lean_fac = state_lean['state_lean_factor_gop'].iloc[0]

            # Determine candidates pcts in the state based on state leans
            state_nat = national_df.copy()
            state_nat['Harris_split'] = state_nat['Harris_ewma_pct'] * dem_lean_fac
            state_nat['Trump_split'] = state_nat['Trump_ewma_pct'] * gop_lean_fac

            state_df = pd.merge(state_nat, df_state_avg, on='mid_date', suffixes=('_nat', '_state'))

            state_df['Harris_state_final'] = (state_df['Harris_ewma_pct_state'] * state_df['state_poll_weight']) + (state_df['Harris_split'] * (1 - state_df['state_poll_weight']))
            state_df['Trump_state_final'] = (state_df['Trump_ewma_pct_state'] * state_df['state_poll_weight']) + (state_df['Trump_split'] * (1 - state_df['state_poll_weight']))
            
            # Add to text files
            upload_df = state_df[['mid_date', 'Harris_state_final', 'Trump_state_final']].copy()
        
        param_val = f"{ca}-{pa}-{w}"    
        upload_df.loc[:, 'Parameters'] = param_val
        state_polls[state] = upload_df
            
                    
    for state in state_polls:
        data_filename = f"/Users/mattg/Personal Projects/2024_forecast/Data/forecast_results/v1/h2h/{state}.txt"
        if not os.path.exists(data_filename):
            with open(data_filename, 'w') as file:
                file.write('Date,Harris_avg,Trump_avg,Parameters\n')

        with open(data_filename, 'a') as file:
                state_polls[state].to_csv(file, header=False, index=False)


print("Running forecast")
start_model = time.time()               
i = 1     
for count_alpha in range(35, 46):
    for pct_alpha in range(50, 86, 2):
        for poll_weight_val in range(15, 36, 2):
            iteration_start = time.time()
            
            ca = count_alpha / 100
            pa = pct_alpha / 100
            w = poll_weight_val / 10
            
            print(f"Percent Alpha: {pa}")
            print(f"Count Alpha:   {ca}")
            print(f"State Weight:  {w}")
            print(f"Completed {i} / 2178")
            i = i + 1
            
            attempts = 0
            max_retries = 10
            while attempts <= max_retries:
                try:
                    model_v1(df, ca, pa, w)
                    break
                except Exception as e:
                    print(f"Error encountered: {e}. Retrying... Attempt {attempts + 1}/{max_retries}")
            
            if attempts == max_retries:
                print(f"Max retries reached for count_alpha={ca}, pct_alpha={pa}, poll_weight_val={w}. Moving to next parameters.")
                
            iteration_end = time.time()
            
            iteration_time = iteration_end - iteration_start
            print(f"Seconds: {iteration_time:.5f}")
            print("----------------------")
                
                
end_model = time.time()
model_time = end_model - start_model
hours, remainder = divmod(model_time, 3600)
minutes, seconds = divmod(remainder, 60)
print(f"The model time was: {int(hours)} hours - {int(minutes)} minutes - {seconds:.2f} seconds")
