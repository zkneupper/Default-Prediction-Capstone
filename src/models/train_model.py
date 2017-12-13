

# Import libraries
import os
import sys

# cpu_count returns the number of CPUs in the system.
from multiprocessing import cpu_count

import numpy as np
import pandas as pd

# Import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

# Import preprocessing methods from sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler

# Import PCA
from sklearn.decomposition import PCA

# Import feature_selection tools
from sklearn.feature_selection import VarianceThreshold

# Import models from sklearn
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

# Import XGBClassifier
from xgboost.sklearn import XGBClassifier

# Import from sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator

# Import plotting libraries
import matplotlib.pyplot as plt

# Modify notebook settings
pd.options.display.max_columns = 150
pd.options.display.max_rows = 150
%matplotlib inline
plt.style.use('ggplot')


# Create a variable for the project root directory
proj_root = os.path.join(os.pardir)

# Save path to the processed data file
# "dataset_processed.csv"
processed_data_file = os.path.join(proj_root,
                                   "data",
                                   "processed",
                                   "dataset_processed.csv")


# add the 'src' directory as one where we can import modules
src_dir = os.path.join(proj_root, "src")
sys.path.append(src_dir)



# Save the path to the folder that will contain 
# the figures for the final report:
# /reports/figures
figures_dir = os.path.join(proj_root,
                                "reports",
                                "figures")


# Read in the processed credit card client default data set.
df = pd.read_csv(processed_data_file, 
                           index_col=0)



#Train test split

# Extract X and y from df
X = df.drop('y', axis=1).values
#y = df[['y']].values
y = df['y'].values

# Train test split
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3, random_state=42)



# Define a function`namestr` to access the name of a variable
def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj][0]



#Make pipeline

df_X = df.drop('y', axis=1)




def create_binary_feature_list(df=df_X,  
                               return_binary_features=True):
    """
    Docstring ...
    """
    # Create boolean maskDrop the column with the target values
    binary_mask = df.isin([0, 1]).all()
    
    # If return_binary_features=True,
    # create a list of the binary features.
    # If return_binary_features=False,
    # create a list of the nonbinary features.    
    features_list = list(binary_mask[binary_mask == \
                                     return_binary_features].index)

    return features_list


def binary_feature_index_list(df=df_X, 
                              features_list=None):
    """
    Docstring ...
    """
    
    feature_index_list = [df.columns.get_loc(c) for c  \
                          in df.columns if c in features_list]    
    
    return feature_index_list


binary_features = create_binary_feature_list(df=df_X, 
                                             return_binary_features=True)

non_binary_features = create_binary_feature_list(df=df_X,  
                                                 return_binary_features=False)

binary_index_list = \
    binary_feature_index_list(df=df_X, 
                              features_list=binary_features)

non_binary_index_list = \
    binary_feature_index_list(df=df_X, 
                              features_list=non_binary_features)



#User defined preprocessors    

class NonBinary_PCA(BaseEstimator, TransformerMixin): 
    def __init__(self):
        self.scaler = PCA(n_components=None, random_state=42)

    # Fit PCA only on the non-binary features
    def fit(self, X, y):
        self.scaler.fit(X[:, non_binary_index_list], y)
        return self

    # Transform only the non-binary features with PCA
    def transform(self, X):
        X_non_binary = \
            self.scaler.transform(X[:, non_binary_index_list])

        X_recombined = X_non_binary

        binary_index_list.sort()
        for col in binary_index_list:
            X_recombined = np.insert(X_recombined, 
                                     col,
                                     X[:, col], 
                                     1)
        return X_recombined



class NonBinary_RobustScaler(BaseEstimator, TransformerMixin): 
    def __init__(self):
        self.scaler = RobustScaler()

    # Fit RobustScaler only on the non-binary features
    def fit(self, X, y):
        self.scaler.fit(X[:, non_binary_index_list], y)
        return self

    # Transform only the non-binary features with RobustScaler
    def transform(self, X):
        X_non_binary = \
            self.scaler.transform(X[:, non_binary_index_list])

        X_recombined = X_non_binary

        binary_index_list.sort()
        for col in binary_index_list:
            X_recombined = np.insert(X_recombined, 
                                     col,
                                     X[:, col], 
                                     1)
        return X_recombined


class NonBinary_StandardScaler(BaseEstimator, TransformerMixin): 
    def __init__(self):
        self.scaler = StandardScaler()

    # Fit StandardScaler only on the non-binary features
    def fit(self, X, y):
        self.scaler.fit(X[:, non_binary_index_list], y)
        return self

    # Transform only the non-binary features with StandardScaler
    def transform(self, X):
        X_non_binary = \
            self.scaler.transform(X[:, non_binary_index_list])

        X_recombined = X_non_binary

        binary_index_list.sort()
        for col in binary_index_list:
            X_recombined = np.insert(X_recombined, 
                                     col,
                                     X[:, col], 
                                     1)
        return X_recombined



class NonBinary_MinMaxScaler(BaseEstimator, TransformerMixin): 
    def __init__(self):
        self.scaler = MinMaxScaler()

    # Fit MinMaxScaler only on the non-binary features
    def fit(self, X, y):
        self.scaler.fit(X[:, non_binary_index_list], y)
        return self

    # Transform only the non-binary features with MinMaxScaler
    def transform(self, X):
        X_non_binary = \
            self.scaler.transform(X[:, non_binary_index_list])

        X_recombined = X_non_binary

        binary_index_list.sort()
        for col in binary_index_list:
            X_recombined = np.insert(X_recombined, 
                                     col,
                                     X[:, col], 
                                     1)
        return X_recombined        



#Define the pipeline

# Set a high threshold for removing near-zero variance features
#thresh_prob = 0.999
thresh_prob = 0.99
threshold = (thresh_prob * (1 - thresh_prob))

# Create pipeline
pipe = Pipeline([('preprocessing_1', VarianceThreshold(threshold)), 
                 ('preprocessing_2', None), 
                 ('preprocessing_3', None), 
                 ('classifier', DummyClassifier(strategy='most_frequent',
                                                random_state=42))])

# Create parameter grid
param_grid = [
    {'classifier': [LogisticRegression(random_state=42)],
     'preprocessing_1': [None, NonBinary_RobustScaler()],
     'preprocessing_2': [None, NonBinary_PCA()],
     'preprocessing_3': [None, VarianceThreshold(threshold)], 
     'classifier__C': [0.01, 0.1],
     'classifier__penalty': ['l1','l2']},

    
    {'classifier': [XGBClassifier(objective='binary:logistic', n_estimators=1000)], 
     'preprocessing_1': [None, VarianceThreshold(threshold)], 
     'preprocessing_2': [None],
     'preprocessing_3': [None],
     'classifier__n_estimators': [1000],
     'classifier__learning_rate': [0.01, 0.1],
     'classifier__gamma': [0.01, 0.1],
     'classifier__max_depth': [3, 4],
     'classifier__min_child_weight': [1, 3],
     'classifier__subsample': [0.8],
#     'classifier__colsample_bytree': [0.8, 1.0],
     'classifier__reg_lambda': [0.1, 1.0],
     'classifier__reg_alpha': [0, 0.1]}]



# Set the number of cores to be used
cores_used = cpu_count() - 1
cores_used
cores_used = 1


# Set verbosity
verbosity = 1

# Execute Grid search
grid = GridSearchCV(pipe, param_grid, cv=5, scoring='roc_auc',
                    verbose=verbosity, n_jobs=cores_used)

grid.fit(X_train, y_train)

print("Best params:\n{}\n".format(grid.best_params_))
print("Best cross-validation score: {:.2f}".format(grid.best_score_))




# Save the grid search object as a pickle file

# Save path to the `models` folder
models_folder = os.path.join(proj_root,
                             "models")


# full_gridsearch_file_name = 'gridsearch_pickle_20171029.pkl'
full_gridsearch_file_name = 'gridsearch_pickle.pkl'

full_gridsearch_path = os.path.join(models_folder,
                                    full_gridsearch_file_name)

joblib.dump(grid, full_gridsearch_path)



# best_pipeline_file_name = 'pipeline_pickle_20171029.pkl'
best_pipeline_file_name = 'pipeline_pickle.pkl'

best_pipeline_path = os.path.join(models_folder, 
                                  best_pipeline_file_name)

joblib.dump(grid.best_estimator_, best_pipeline_path)




















