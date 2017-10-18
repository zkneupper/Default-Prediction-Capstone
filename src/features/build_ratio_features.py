
# Written by Zachary Kneupper
# 2017-10-16

# This program was written to take the wrangled/cleaned 
# data in 'dataset_wrangled.csv', engineer new features,
# and to export the new data set with these new features
# to a new file called 'dataset_interim.csv'

import os
import sys
import pandas as pd


def build_features_ba_over_cl(df):
    """
    This function takes the wrangled data DataFrame as its argument,
    calculates the ratio of (bill amount / credit limit) for each of
    the six BILL_AMT features, and returns a new DataFrame that 
    includes the six new (bill amount / credit limit) features, 
    in addition to the original wrangled data DataFrame.
    """

    bill_amount_column_list = ['BILL_AMT1',
                               'BILL_AMT2',
                               'BILL_AMT3',
                               'BILL_AMT4',
                               'BILL_AMT5',
                               'BILL_AMT6']
    
    for i, ba in enumerate(bill_amount_column_list, 1):

        new_column_name = 'ba_over_cl_' + str(i)
        
        df[new_column_name] = df[ba] / df['LIMIT_BAL']

    return df


def build_features_ba_less_pa_over_cl(df):
    """
    This function takes the wrangled data DataFrame as its argument,
    calculates the ratio of ((bill amount−pay amount)/credit limit)
    for each of the six pairs of BILL_AMT and PAY_AMT features, and 
    returns a new DataFrame that includes the six new 
    ((bill amount−pay amount)/credit limit) features, 
    in addition to the original wrangled data DataFrame.
    """

    bill_amount_column_list = ['BILL_AMT1',
                               'BILL_AMT2',
                               'BILL_AMT3',
                               'BILL_AMT4',
                               'BILL_AMT5',
                               'BILL_AMT6']

    pay_amount_column_list = ['PAY_AMT1',
                              'PAY_AMT2',
                              'PAY_AMT3',
                              'PAY_AMT4',
                              'PAY_AMT5',
                              'PAY_AMT6'] 
    
    for i, (ba, pa) in enumerate(zip(bill_amount_column_list, 
                               pay_amount_column_list), 
                           1):

        new_column_name = 'ba_less_pa_over_cl_' + str(i)
        
        df[new_column_name] = \
            (df[ba] -df[pa])/ df['LIMIT_BAL']

    return df


def create_interim_dataset(new_file_name='dataset_interim.csv'):

    # Create a variable for the project root directory
    proj_root = os.path.join(os.pardir)

    # Save path to the wrangled data file
    # "dataset_wrangled.csv"
    wrangled_data_file = os.path.join(proj_root,
                                    "data",
                                    "interim",
                                    "dataset_wrangled.csv")

    # Save the path to the folder that will contain 
    # the new interim data set:
    # /data/interim
    interim_data_dir = os.path.join(proj_root,
                                    "data",
                                    "interim")

    # Read in the wrangled credit card client default data set.
    df = pd.read_csv(wrangled_data_file,
                              header=1, 
                              index_col=0)

    df = build_features_ba_over_cl(df)

    df = build_features_ba_less_pa_over_cl(df)

    new_file_path = os.path.join(interim_data_dir,
                                 new_file_name)

    df.to_csv(new_file_path)
