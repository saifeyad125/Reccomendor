import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# File paths
file_path_1 = '/Users/saif/Desktop/University Saif/aldar internship/vscodealdar/movierecommend/archive/tmdb_5000_movies.csv'
file_path_2 = '/Users/saif/Desktop/University Saif/aldar internship/vscodealdar/movierecommend/archive/tmdb_5000_credits.csv'

# Load the data
def load_data():
    """Load and merge the movie and credits datasets."""
    # Read CSV files
    dataframe = pd.read_csv(file_path_1)
    dataframe2 = pd.read_csv(file_path_2)

    # Rename columns in dataframe2 to match dataframe
    dataframe2.columns = ['id', 'tittle', 'cast', 'crew']

    # Merge the two dataframes on the 'id' column
    merged_data = dataframe.merge(dataframe2, on='id')

    return merged_data

__all__ = ['pd', 'np', 'plt', 'load_data']

