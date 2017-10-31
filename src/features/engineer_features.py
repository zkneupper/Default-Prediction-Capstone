
# Written by Zachary Kneupper
# 2017-10-18

# This program was written to take the interim
# data in 'dataset_interim.csv', 
# move the target variable to the leftmost column,
# rename columns where necessary,
# one-hot-encode categorical variables, and
# and to export the new data set with these new features
# to a new file called 'dataset_processed.csv'

import os
import sys
import pandas as pd
import numpy as np


def create_processed_dataset(new_file_name='dataset_processed.csv'):

    # Create a variable for the project root directory
    proj_root = os.path.join(os.pardir)

    # Save path to the interim data file
    # "dataset_interim.csv"
    interim_data_file = os.path.join(proj_root,
                                    "data",
                                    "interim",
                                    "dataset_interim.csv")

    # Save the path to the folder that will contain 
    # the final, canonical data sets for modeling:
    # /data/processed
    processed_data_dir = os.path.join(proj_root,
                                    "data",
                                    "processed")

    # Read in the interim credit card client default data set.
    df_interim = pd.read_csv(interim_data_file,
                             index_col=0)

    # Get the current list of columns names
    col_names = list(df_interim.columns)

    # Identify the target variable column name
    target = 'default payment next month'

    # Move the target column name to the beginning
    # of the list using index, pop, and insert
    col_names.insert(0, col_names.pop(col_names.index(target)))

    # Use the reordered list of columns names 
    # and .loc to reorder the columns in df_interim
    df_interim = df_interim.loc[:, col_names]

    # Get the current list of columns names
    col_names_current = list(df_interim.columns)

    # Replace capital letters with lower-case letters 
    # and replace spaces with underscores
    col_names_new = [s.lower().replace(" ", "_") for s in col_names_current]

    cols_dict = {old: new for (old, new) in zip(col_names_current,
                                                col_names_new)}

    df_interim.rename(columns=cols_dict, inplace=True)


    # Get the current list of columns names
    col_names_current = list(df_interim.columns)

    # Rename columns that have long names
    # Make a dictionary to rename columns that have long names,
    # where the keys are the current names and the 
    # values are the new, shorter names.
    cols_dict = {'default_payment_next_month': 'y',
                 'education': 'edu',
                 'ba_over_cl_1': 'bl_ratio_1',
                 'ba_over_cl_2': 'bl_ratio_2',
                 'ba_over_cl_3': 'bl_ratio_3',
                 'ba_over_cl_4': 'bl_ratio_4',
                 'ba_over_cl_5': 'bl_ratio_5',
                 'ba_over_cl_6': 'bl_ratio_6',
                 'ba_less_pa_over_cl_1': 'blpl_ratio_1',
                 'ba_less_pa_over_cl_2': 'blpl_ratio_2',
                 'ba_less_pa_over_cl_3': 'blpl_ratio_3',
                 'ba_less_pa_over_cl_4': 'blpl_ratio_4',
                 'ba_less_pa_over_cl_5': 'blpl_ratio_5',
                 'ba_less_pa_over_cl_6': 'blpl_ratio_6'}

    df_interim.rename(columns=cols_dict, inplace=True)

    # One-hot-encode categorical variables
    # Make a list of categorical columns
    categorical_vars = ['sex', 'edu', 'marriage', 
                        'pay_1', 'pay_2', 'pay_3', 
                        'pay_4', 'pay_5', 'pay_6']

    # Cast values in the categorical columns as type str.
    df_interim[categorical_vars] = df_interim[categorical_vars].astype(str)

    # One-hot-encode the categorical variables
    df_interim = pd.get_dummies(df_interim, 
                                columns=categorical_vars, 
                                drop_first=True)

    new_file_path = os.path.join(processed_data_dir,
                                 new_file_name)

    df_interim.to_csv(new_file_path)
