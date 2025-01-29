import pandas as pd
import numpy as np

file_path_1 = '/Users/saif/Desktop/University Saif/aldar internship/vscodealdar/archive/tmdb_5000_movies.csv'
file_path_2 = '/Users/saif/Desktop/University Saif/aldar internship/vscodealdar/archive/tmdb_5000_credits.csv'

dataframe = pd.read_csv(file_path_1)
dataframe2 = pd.read_csv(file_path_2)

#first i want to merge the two dataframes:

#i will rename the colums of df2 to match df 1:
dataframe2.columns = ['id','tittle','cast','crew']
dataframe = dataframe.merge(dataframe2, on='id')


print(dataframe.head(5))
