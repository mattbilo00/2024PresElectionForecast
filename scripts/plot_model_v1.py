import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

### Plot vote of each state over time ###
PATH = "/Users/mattg/Personal Projects/2024_forecast/Data/forecast_results/v1/h2h"
current_pct = pd.DataFrame(columns=['StateName', 'harris_pct', 'trump_pct', 'diff'])

date_dict = {
    '2024-07-13': ' First assassination \n attempt on Donald Trump',
    '2024-07-15': ' RNC begins',
    '2024-08-19': ' DNC begins',
    '2024-09-15': ' Second assassination attempt on Donald Trump',
    '2024-09-10': ' Presidential \n Debate',
    '2024-07-21': ' Joe Biden drops out of race'
}

for filename in os.listdir(PATH):
    if filename.endswith('.txt'):
        file_path = os.path.join(PATH, filename)
        df = pd.read_csv(file_path)
        state = filename.replace('.txt', '')

        # Group by 'Date' and aggregate
        df = df.groupby('Date').agg(
            Harris_mean=('Harris_avg', 'mean'),
            Trump_mean=('Trump_avg', 'mean'),
            Harris_min=('Harris_avg', 'min'),
            Harris_max=('Harris_avg', 'max'),
            Trump_min=('Trump_avg', 'min'),
            Trump_max=('Trump_avg', 'max')
        ).reset_index()

        # Strip whitespace from column names
        df.columns = df.columns.str.strip()

        # Convert 'Date' to datetime format
        df['Date'] = pd.to_datetime(df['Date'])

        # Calculate moving averages
        window = 5
        df['Harris_mean_smoothed'] = df['Harris_mean'].rolling(window=window).mean()
        df['Trump_mean_smoothed'] = df['Trump_mean'].rolling(window=window).mean()

        # Set up the plot
        fig, ax = plt.subplots(figsize=(12, 8))

        # Set the background color of the figure and axes
        fig.patch.set_facecolor('#f0f0f0')
        ax.set_facecolor('#f0f0f0')

        # Plot Harris and Trump Mean (smoothed) without markers and with rounded lines
        ax.plot(df['Date'], df['Harris_mean_smoothed'], color='blue', label=f'Harris {window} Day Average', linewidth=2, solid_capstyle='round')
        ax.plot(df['Date'], df['Trump_mean_smoothed'], color='red', label=f'Trump {window} Day Average', linewidth=2, solid_capstyle='round')

        # Add error bands for mean
        ax.fill_between(df['Date'], df['Harris_min'], df['Harris_max'], color='blue', alpha=0.15, label='Harris model output')
        ax.fill_between(df['Date'], df['Trump_min'], df['Trump_max'], color='red', alpha=0.15, label='Trump model output')

        # Customize plot appearance
        ax.set_title(f'{state.upper()}')
        ax.set_xlabel('Date')
        ax.set_ylim(20, 80)
        ax.legend(loc='lower left')
        ax.grid()

        # Get the most recent smoothed values
        recent_harris = df['Harris_mean_smoothed'].iloc[-1]
        recent_trump = df['Trump_mean_smoothed'].iloc[-1]
        
        new_row = {'StateName': state, 'harris_pct': recent_harris, 'trump_pct': recent_trump, 'diff': recent_harris - recent_trump}
        current_pct.loc[len(current_pct)] = new_row

        # Add text boxes with the most recent values
        harris_textstr = f'Harris: {recent_harris:.2f}'
        trump_textstr = f'Trump: {recent_trump:.2f}'

        # Harris box properties
        harris_props = dict(boxstyle='round', facecolor=(0, 0, 1, 0.2), edgecolor='blue', linewidth=2)  # Use tuple for RGBA
        
        # Trump box properties
        trump_props = dict(boxstyle='round', facecolor=(1, 0, 0, 0.2), edgecolor='red', linewidth=2)  # Use tuple for RGBA
        
        # Position text boxes based on values
        if recent_harris > recent_trump:
            # Harris value is larger, place Harris at the top right
            ax.text(0.95, 0.93, harris_textstr, transform=ax.transAxes, fontsize=14,
                    verticalalignment='top', horizontalalignment='right', bbox=harris_props)
            # Trump at the bottom right
            ax.text(0.95, 0.13, trump_textstr, transform=ax.transAxes, fontsize=14,
                    verticalalignment='top', horizontalalignment='right', bbox=trump_props)
        else:
            # Trump value is larger, place Trump at the top right
            ax.text(0.95, 0.93, trump_textstr, transform=ax.transAxes, fontsize=14,
                    verticalalignment='top', horizontalalignment='right', bbox=trump_props)
            # Harris at the bottom right
            ax.text(0.95, 0.10, harris_textstr, transform=ax.transAxes, fontsize=14,
                    verticalalignment='top', horizontalalignment='right', bbox=harris_props)

        # Adjust layout and show plot
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust the title to fit
        plt.savefig(f"/Users/mattg/Personal Projects/2024_forecast/web page/v1_images/{state}.png", dpi=300)



### Plot EC votes ###

# Suppress only the findfont warnings
warnings.filterwarnings("ignore", category=UserWarning, module='matplotlib')

# Set the font to Rubik
rcParams['font.family'] = 'Rubik'

# Load the data
ec_df = pd.read_excel("/Users/mattg/Personal Projects/2024_forecast/Data/electoral_votes.xlsx")
current_pct['StateName'] = current_pct['StateName'].str.lower()
ec_df['StateName'] = ec_df['StateName'].str.lower()

# Merge DataFrames
df = pd.merge(current_pct, ec_df, on='StateName')

# Sort the DataFrame by the absolute value of diff
df['abs_diff'] = df['diff'].abs()  # Create a new column for absolute diff
df = df.sort_values(by='abs_diff')  # Sort by absolute diff

# Set up grid layout for plotting
n_cols = 7  # Number of columns
n_rows = 8  # Number of rows

# Create y-coordinates based on grid layout and flip the order
y_positions = [n_rows - 1 - (i // n_cols) for i in range(len(df))]  # Flip y-coordinates
x_positions = [i % n_cols for i in range(len(df))]  # x-values in columns

# Set up the color based on the new logic for abs_diff
colors = []
for val in df['diff']:
    abs_diff = abs(val)
    if abs_diff < 3:
        colors.append((0.5, 0.5, 0.5, 1))  # Grey for abs_diff < 3
    elif abs_diff < 10:
        if val > 0:
            colors.append((0.7, 0.7, 1, 1))  # Light blue for positive diff, abs_diff < 10
        else:
            colors.append((1, 0.7, 0.7, 1))  # Light red for negative diff, abs_diff < 10
    elif abs_diff < 20:
        if val > 0:
            colors.append((0.4, 0.4, 1, 1))  # Darker blue for positive diff, abs_diff < 20
        else:
            colors.append((1, 0.4, 0.4, 1))  # Darker red for negative diff, abs_diff < 20
    else:
        if val > 0:
            colors.append((0, 0, 1, 1))  # Darkest blue for positive diff, abs_diff >= 20
        else:
            colors.append((1, 0, 0, 1))  # Darkest red for negative diff, abs_diff >= 20

# Create the plot
plt.figure(figsize=(12, 10))

# Set the background color of the figure
plt.gcf().set_facecolor('#f0f0f0')  # Set the figure background color

# Scatter plot
plt.scatter(x_positions, y_positions,  # x and y positions for the scatter plot
            s=df['EC'] * 300,  # Size based on EC (scaled up more for visibility)
            c=colors,  # Use the color list created earlier
            alpha=0.6,  # Set transparency
            edgecolors='w')  # Add white edge color

# Add annotations (centered on the circles)
for idx, row in df.iterrows():
    pos_idx = df.index.get_loc(idx)  # Use position in sorted DataFrame
    rounded_diff = round(row['diff'], 1)  # Round diff to one decimal place
    
    # Determine the label based on the diff value
    if rounded_diff > 0:
        label_diff = f"H+{rounded_diff}"  # Positive values get H+
    elif rounded_diff < 0:
        label_diff = f"T+{abs(rounded_diff)}"  # Negative values get T+
    else:
        label_diff = "EVEN"  # Replace 0.0 with EVEN
    
    # Add the text with the new format
    plt.text(x_positions[pos_idx], y_positions[pos_idx], 
            f"{row['StateAbbv']}\n{row['EC']}\n{label_diff}", 
            ha='center', va='center', fontsize=12, fontweight='bold', color='black')

# Remove titles, labels, borders, and grid lines
plt.gca().set_frame_on(False)  # Remove the plot border
plt.gca().axis('off')  # Turn off axis labels, ticks, and grid lines

# Set x and y limits
plt.xlim(-1, n_cols)  # Set x limits slightly wider
plt.ylim(0, n_rows)  # Set y limits slightly wider, allowing space for text

plt.tight_layout()  # Automatically adjust subplot parameters for better spacing
plt.savefig(f"/Users/mattg/Personal Projects/2024_forecast/web page/v1_images/ec_map.png", dpi=300)
