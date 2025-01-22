import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk


def show_dataframe_popup(df):
    root = tk.Tk()
    root.title("DataFrame Viewer")
    
    frame = ttk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True)

    # Create Treeview widget
    tree = ttk.Treeview(frame, columns=list(df.columns), show="headings", height=15)

    # Add column headings
    for col in df.columns:
        tree.heading(col, text=col)
        tree.column(col, width=150, anchor='center')

    # Add rows to the Treeview
    for index, row in df.iterrows():
        tree.insert("", "end", values=list(row))

    tree.pack(fill=tk.BOTH, expand=True)
    root.mainloop()


file_path_1 = '/Users/saif/Desktop/University Saif/aldar internship/vscodealdar/googleplaystore/archive/googleplaystore_user_reviews.csv'
file_path_2 = '/Users/saif/Desktop/University Saif/aldar internship/vscodealdar/googleplaystore/archive/googleplaystore.csv'

def load_data():
    dataframe1 = pd.read_csv(file_path_1)
    dataframe2 = pd.read_csv(file_path_2)

    merged_data = dataframe1.merge(dataframe2, on='App')
    
    return merged_data

__all__ = ['pd', 'np', 'plt', 'load_data', 'show_dataframe_popup', 'tk', 'ttk']

