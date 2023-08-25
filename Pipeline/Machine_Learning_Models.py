#!/usr/bin/env python
# coding: utf-8

# In[3]:


import subprocess
def install(package, version):

    """
    Install or upgrade a specific version of a package using pip.

    Parameters:
    - package (str): Name of the package to install.
    - version (str): Version of the package to install.

    Returns:
    None
    """

    subprocess.call(["pip", "install", "--upgrade", f"{package}=={version}"])

install("scikit-learn", "1.2.2")


# In[4]:


libraries_to_install = ['xgboost','tensorflow', 'keras-tuner']

for library in libraries_to_install:
    subprocess.run(["pip", "install", library], check=True)


# In[7]:


import Data_Processing as dp
from pandas_datareader import data as pdr
from sklearn.preprocessing import StandardScaler
from scipy import stats
import pandas as pd
from datetime import date
import numpy as np
import math
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, balanced_accuracy_score
from datetime import date
import glob, os
from joblib import dump, load
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import ReduceLROnPlateau
import logging; tf.get_logger().setLevel(logging.ERROR)
from keras import Model, layers, models, optimizers, losses, metrics
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.losses import *
from keras.metrics import *
import time
import sys
import keras_tuner
from keras_tuner import Hyperband
import warnings
warnings.filterwarnings("ignore")
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from operator import concat


# # 1. SVM

# In[12]:


def SVM_Gaussian(ticker = 'USDEUR', feature_dic = None, upload = False, path = '/FX-Data/',
                         train_start = '2005-01-03', train_end = '2011-12-30',
                         vali1_start = '2012-01-02', vali1_end = '2014-12-31',
                         vali2_start = '2015-01-01', vali2_end = '2017-12-29',
                         test_start = '2018-01-01', test_end = '2018-12-31',
                         standarlization = True, from_beginning = True):

    """
    Train and evaluate an SVM with Gaussian Kernel for a given ticker.
    
    Parameters:
    - ticker (str): Ticker symbol, default is 'USDEUR'.
    - feature_dic (dict or None): Dictionary of features for the given ticker. If None, no feature selection is done.
    - upload (bool): Whether to load a pre-trained model or not.
    - path (str): Path to data, default is '/FX-Data/'.
    - train_start (str): Start date for the training dataset.
    - train_end (str): End date for the training dataset.
    - vali1_start (str): Start date for the first validation dataset.
    - vali1_end (str): End date for the first validation dataset.
    - vali2_start (str): Start date for the second validation dataset.
    - vali2_end (str): End date for the second validation dataset.
    - test_start (str): Start date for the testing dataset.
    - test_end (str): End date for the testing dataset.
    - standarlization (bool): Whether to standardize data or not.
    - from_beginning (bool): If true, use data from the beginning.
    
    Returns:
    - test_score (float): Accuracy score on the test data.
    - balanced_score1 (float): Balanced accuracy score on the test data.
    """

    if feature_dic is not None and not isinstance(feature_dic, dict):
        raise ValueError("feature_dic must be of type dict or None.")

    # Data split
    X_train, X_val1, X_val2, X_test, y_train, y_val1, y_val2, y_test = dp.getProcessedData(ticker, path,
                                                                             train_start, train_end,
                                                                             vali1_start, vali1_end,
                                                                             vali2_start, vali2_end,
                                                                             test_start, test_end,
                                                                             standarlization, from_beginning)
    working_path = os.getcwd()
    if feature_dic != None:
        # For each market, we only use several top features selected by SFFS.
        X_train = X_train.iloc[:,feature_dic[ticker]]
        X_val1 = X_val1.iloc[:,feature_dic[ticker]]
        X_val2 = X_val2.iloc[:,feature_dic[ticker]]
        X_test = X_test.iloc[:,feature_dic[ticker]]

    # Check if the model should be loaded directly
    if upload == True:
        # For ensure that the model does not change when it is used again and is easy to reproduce.
        svm = load(working_path + '/model_weights/' + ticker +'svmnon.h5')
        y_pred = svm.predict(X_val2)
        # Calculating the accuracy score
        test_score = svm.score(X_val2,y_val2)
        # Calculating the balanced accuracy score
        balanced_score1 = balanced_accuracy_score(y_val2, y_pred)
    # Otherwise, train the model from scratch
    elif upload == False:
        best_score = 0

        # Grid search for best parameters
        for gamma in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]:
            for C in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]:
                svm = SVC(gamma=gamma, C=C,kernel = 'rbf')
                svm.fit(X_train, y_train)
                score = svm.score(X_val1, y_val1)
                if score > best_score:
                    best_score = score
                    best_parameters = {'C': C, 'gamma': gamma}
        C1 = best_parameters['C']
        gamma1 = best_parameters['gamma']
        # Training the SVM model with best parameters
        svm = SVC(gamma=gamma1, C=C1,kernel = 'rbf')
        svm.fit(X_train, y_train)

        # Save as a h5 format file
        path1 = working_path + '/model_weights/' + ticker +'svmnon.h5'
        os.makedirs(os.path.dirname(path1), exist_ok=True)
        dump(svm, path1)

        y_pred = svm.predict(X_val2)
        # Calculating the accuracy score
        test_score = svm.score(X_val2,y_val2)
        # Calculating the balanced accuracy score
        balanced_score1 = balanced_accuracy_score(y_val2, y_pred)

    return test_score, balanced_score1


# # 2. LR

# In[13]:


def LR_Lasso(ticker = 'USDEUR', feature_dic = None, upload = False, path = '/FX-Data/',
                         train_start = '2005-01-03', train_end = '2011-12-30',
                         vali1_start = '2012-01-02', vali1_end = '2014-12-31',
                         vali2_start = '2015-01-01', vali2_end = '2017-12-29',
                         test_start = '2018-01-01', test_end = '2018-12-31',
                         standarlization = True, from_beginning = True):

    """
    Train and evaluate a Logistic Regression with Lasso regularization for a given ticker.
    
    Parameters:
    - ticker (str): Ticker symbol, default is 'USDEUR'.
    - feature_dic (dict or None): Dictionary of features for the given ticker. If None, no feature selection is done.
    - upload (bool): Whether to load a pre-trained model or not.
    - path (str): Path to data, default is '/FX-Data/'.
    - train_start (str): Start date for the training dataset.
    - train_end (str): End date for the training dataset.
    - vali1_start (str): Start date for the first validation dataset.
    - vali1_end (str): End date for the first validation dataset.
    - vali2_start (str): Start date for the second validation dataset.
    - vali2_end (str): End date for the second validation dataset.
    - test_start (str): Start date for the testing dataset.
    - test_end (str): End date for the testing dataset.
    - standarlization (bool): Whether to standardize data or not.
    - from_beginning (bool): If true, use data from the beginning.
    
    Returns:
    - test_score (float): Accuracy score on the test data.
    - balanced_score1 (float): Balanced accuracy score on the test data.
    """

    if feature_dic is not None and not isinstance(feature_dic, dict):
        raise ValueError("feature_dic must be of type dict or None.")

    # Data split
    X_train, X_val1, X_val2, X_test, y_train, y_val1, y_val2, y_test = dp.getProcessedData(ticker, path,
                                                                             train_start, train_end,
                                                                             vali1_start, vali1_end,
                                                                             vali2_start, vali2_end,
                                                                             test_start, test_end,
                                                                             standarlization, from_beginning)
    working_path = os.getcwd()
    if feature_dic != None:
        # For each market, we only use several top features selected by SFFS.
        X_train = X_train.iloc[:,feature_dic[ticker]]
        X_val1 = X_val1.iloc[:,feature_dic[ticker]]
        X_val2 = X_val2.iloc[:,feature_dic[ticker]]
        X_test = X_test.iloc[:,feature_dic[ticker]]

    # Check if the model should be loaded directly
    if upload == True:
        # For ensure that the model does not change when it is used again and is easy to reproduce.
        lg1 = load(working_path + '/model_weights/' + ticker + 'lr.h5')
        # Calculating the accuracy score
        y_pred = lg1.predict(X_val2)
        # Calculating the accuracy score
        test_score = lg1.score(X_val2,y_val2)
        # Calculating the balanced accuracy score
        balanced_score1 = balanced_accuracy_score(y_val2, y_pred)
    # Otherwise, train the model from scratch
    elif upload == False:
        lg = LogisticRegression()
        best_score = 0

        for C in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]:
            lg = LogisticRegression(penalty='l1', solver='saga', C=C ,max_iter=8000)
            lg.fit(X_train, y_train)
            score = lg.score(X_val1, y_val1)
            if score > best_score:
                best_score = score
                best_parameters = {'C': C}
        C1 = best_parameters['C']
        lg1 = LogisticRegression(penalty='l1', solver='saga', C=C1 ,max_iter=8000)
        lg1.fit(X_train, y_train)

        path1 = working_path + '/model_weights/' + ticker + 'lr.h5'
        os.makedirs(os.path.dirname(path1), exist_ok=True)
        dump(lg1, path1)

        y_pred = lg1.predict(X_val2)
        # Calculating the accuracy score
        test_score = lg1.score(X_val2,y_val2)
        # Calculating the balanced accuracy score
        balanced_score1 = balanced_accuracy_score(y_val2, y_pred)

    return test_score, balanced_score1


# # 3.LDA

# In[14]:


def LDA(ticker = 'USDEUR', feature_dic = None, upload = False, path = '/FX-Data/',
                         train_start = '2005-01-03', train_end = '2011-12-30',
                         vali1_start = '2012-01-02', vali1_end = '2014-12-31',
                         vali2_start = '2015-01-01', vali2_end = '2017-12-29',
                         test_start = '2018-01-01', test_end = '2018-12-31',
                         standarlization = True, from_beginning = True):
                        
    """
    Train and evaluate a Linear Discriminant Analysis for a given ticker.
    
    Parameters:
    - ticker (str): Ticker symbol, default is 'USDEUR'.
    - feature_dic (dict or None): Dictionary of features for the given ticker. If None, no feature selection is done.
    - upload (bool): Whether to load a pre-trained model or not.
    - path (str): Path to data, default is '/FX-Data/'.
    - train_start (str): Start date for the training dataset.
    - train_end (str): End date for the training dataset.
    - vali1_start (str): Start date for the first validation dataset.
    - vali1_end (str): End date for the first validation dataset.
    - vali2_start (str): Start date for the second validation dataset.
    - vali2_end (str): End date for the second validation dataset.
    - test_start (str): Start date for the testing dataset.
    - test_end (str): End date for the testing dataset.
    - standarlization (bool): Whether to standardize data or not.
    - from_beginning (bool): If true, use data from the beginning.
    
    Returns:
    - test_score (float): Accuracy score on the test data.
    - balanced_score1 (float): Balanced accuracy score on the test data.
    """

    if feature_dic is not None and not isinstance(feature_dic, dict):
        raise ValueError("feature_dic must be of type dict or None.")

    # Data split
    X_train, X_val1, X_val2, X_test, y_train, y_val1, y_val2, y_test = dp.getProcessedData(ticker, path,
                                                                             train_start, train_end,
                                                                             vali1_start, vali1_end,
                                                                             vali2_start, vali2_end,
                                                                             test_start, test_end,
                                                                             standarlization, from_beginning)
    working_path = os.getcwd()
    if feature_dic != None:
        # For each market, we only use several top features selected by SFFS.
        X_train = X_train.iloc[:,feature_dic[ticker]]
        X_val1 = X_val1.iloc[:,feature_dic[ticker]]
        X_val2 = X_val2.iloc[:,feature_dic[ticker]]
        X_test = X_test.iloc[:,feature_dic[ticker]]

    # Check if the model should be loaded directly
    if upload == True:
        # For ensure that the model does not change when it is used again and is easy to reproduce.
        lda = load(working_path + '/model_weights/' + ticker + 'lda.h5')
        y_pred = lda.predict(X_val2)
        # Calculating the accuracy score
        test_score = lda.score(X_val2,y_val2)
        # Calculating the balanced accuracy score
        balanced_score1 = balanced_accuracy_score(y_val2, y_pred)
    # Otherwise, train the model from scratch
    elif upload == False:
        lda = LinearDiscriminantAnalysis()
        X_train1 = pd.concat([X_train, X_val1],axis=0)
        y_train1 = pd.concat([y_train, y_val1],axis=0)
        lda.fit(X_train1, y_train1)

        path1 = working_path + '/model_weights/' + ticker + 'lda.h5'
        os.makedirs(os.path.dirname(path1), exist_ok=True)
        dump(lda, path1)

        y_pred = lda.predict(X_val2)
        # Calculating the accuracy score
        test_score = lda.score(X_val2,y_val2)
        # Calculating the balanced accuracy score
        balanced_score1 = balanced_accuracy_score(y_val2, y_pred)

    return test_score, balanced_score1


# # 4.QDA

# ### might be a bit different in different computer and python environment, due to the calculation of the float

# In[18]:


def QDA(ticker = 'USDEUR', feature_dic = None, upload = False, path = '/FX-Data/',
                         train_start = '2005-01-03', train_end = '2011-12-30',
                         vali1_start = '2012-01-02', vali1_end = '2014-12-31',
                         vali2_start = '2015-01-01', vali2_end = '2017-12-29',
                         test_start = '2018-01-01', test_end = '2018-12-31',
                         standarlization = True, from_beginning = True):

    """
    Train and evaluate a Quadratic Discriminant Analysis for a given ticker.
    
    Parameters:
    - ticker (str): Ticker symbol, default is 'USDEUR'.
    - feature_dic (dict or None): Dictionary of features for the given ticker. If None, no feature selection is done.
    - upload (bool): Whether to load a pre-trained model or not.
    - path (str): Path to data, default is '/FX-Data/'.
    - train_start (str): Start date for the training dataset.
    - train_end (str): End date for the training dataset.
    - vali1_start (str): Start date for the first validation dataset.
    - vali1_end (str): End date for the first validation dataset.
    - vali2_start (str): Start date for the second validation dataset.
    - vali2_end (str): End date for the second validation dataset.
    - test_start (str): Start date for the testing dataset.
    - test_end (str): End date for the testing dataset.
    - standarlization (bool): Whether to standardize data or not.
    - from_beginning (bool): If true, use data from the beginning.
    
    Returns:
    - test_score (float): Accuracy score on the test data.
    - balanced_score1 (float): Balanced accuracy score on the test data.
    """

    if feature_dic is not None and not isinstance(feature_dic, dict):
        raise ValueError("feature_dic must be of type dict or None.")

    # Data split
    X_train, X_val1, X_val2, X_test, y_train, y_val1, y_val2, y_test = dp.getProcessedData(ticker, path,
                                                                             train_start, train_end,
                                                                             vali1_start, vali1_end,
                                                                             vali2_start, vali2_end,
                                                                             test_start, test_end,
                                                                             standarlization, from_beginning)
    working_path = os.getcwd()
    if feature_dic != None:
        # For each market, we only use several top features selected by SFFS.
        X_train = X_train.iloc[:,feature_dic[ticker]]
        X_val1 = X_val1.iloc[:,feature_dic[ticker]]
        X_val2 = X_val2.iloc[:,feature_dic[ticker]]
        X_test = X_test.iloc[:,feature_dic[ticker]]

    # Check if the model should be loaded directly
    if upload == True:
        # For ensure that the model does not change when it is used again and is easy to reproduce.
        qda = load(working_path + '/model_weights/' + ticker + 'qda.h5')
        y_pred = qda.predict(X_val2)
        # Calculating the accuracy score
        test_score = qda.score(X_val2,y_val2)
        # Calculating the balanced accuracy score
        balanced_score1 = balanced_accuracy_score(y_val2, y_pred)
    # Otherwise, train the model from scratch
    elif upload == False:
        qda = QuadraticDiscriminantAnalysis()
        X_train1 = pd.concat([X_train, X_val1],axis=0)
        y_train1 = pd.concat([y_train, y_val1],axis=0)
        qda.fit(X_train1, y_train1)

        path1 = working_path + '/model_weights/' + ticker + 'qda.h5'
        os.makedirs(os.path.dirname(path1), exist_ok=True)
        dump(qda, path1)

        y_pred = qda.predict(X_val2)
        # Calculating the accuracy score
        test_score = qda.score(X_val2,y_val2)
        # Calculating the balanced accuracy score
        balanced_score1 = balanced_accuracy_score(y_val2, y_pred)

    return test_score, balanced_score1


# # 5.Random Forest

# ### might be a bit different in different computer and python environment, due to the calculation of the float

# In[42]:


def Random_Forest(ticker = 'USDEUR', feature_dic = None, upload = False, path = '/FX-Data/',
                         train_start = '2005-01-03', train_end = '2011-12-30',
                         vali1_start = '2012-01-02', vali1_end = '2014-12-31',
                         vali2_start = '2015-01-01', vali2_end = '2017-12-29',
                         test_start = '2018-01-01', test_end = '2018-12-31',
                         standarlization = True, from_beginning = True, Max_depth = [3,5,7,9],
                         Min_samples_leaf = [1, 2, 4], Min_samples_split = [2, 5, 10]):

    """
    Trains and validates a Random Forest model for a given ticker.

    Parameters:
    - ticker (str): Ticker symbol, default is 'USDEUR'.
    - feature_dic (dict or None): Dictionary of features for the given ticker. If None, no feature selection is done.
    - upload (bool): Whether to load a pre-trained model or not.
    - path (str): Path to data, default is '/FX-Data/'.
    - train_start (str): Start date for the training dataset.
    - train_end (str): End date for the training dataset.
    - vali1_start (str): Start date for the first validation dataset.
    - vali1_end (str): End date for the first validation dataset.
    - vali2_start (str): Start date for the second validation dataset.
    - vali2_end (str): End date for the second validation dataset.
    - test_start (str): Start date for the testing dataset.
    - test_end (str): End date for the testing dataset.
    - standarlization (bool): Whether to standardize data or not.
    - from_beginning (bool): If true, use data from the beginning.
    - Max_depth (list): List of the values of the Maximum depth of the tree that will be considered in the model.
    - Min_samples_leaf (list): List of the values of the Minimum number of samples required at the leaf node that will be considered in the model.
    - Min_samples_split (list): List of the values of the Minimum number of samples required to split an internal node that will be considered in the model.

    Returns:
    - test_score (float): Accuracy score on the test data.
    - balanced_score1 (float): Balanced accuracy score on the test data.
    """

    if feature_dic is not None and not isinstance(feature_dic, dict):
        raise ValueError("feature_dic must be of type dict or None.")

    # Data split
    X_train, X_val1, X_val2, X_test, y_train, y_val1, y_val2, y_test = dp.getProcessedData(ticker, path,
                                                                             train_start, train_end,
                                                                             vali1_start, vali1_end,
                                                                             vali2_start, vali2_end,
                                                                             test_start, test_end,
                                                                             standarlization, from_beginning)
    working_path = os.getcwd()
    if feature_dic != None:
        # For each market, we only use several top features selected by SFFS.
        X_train = X_train.iloc[:,feature_dic[ticker]]
        X_val1 = X_val1.iloc[:,feature_dic[ticker]]
        X_val2 = X_val2.iloc[:,feature_dic[ticker]]
        X_test = X_test.iloc[:,feature_dic[ticker]]

    # Check if the model should be loaded directly
    if upload == True:
        # For ensure that the model does not change when it is used again and is easy to reproduce.
        rf1 = load(working_path + '/model_weights/' + ticker + 'rf.h5')
        y_pred = rf1.predict(X_val2)
        # Calculating the accuracy score
        test_score = rf1.score(X_val2,y_val2)
        # Calculating the balanced accuracy score
        balanced_score1 = balanced_accuracy_score(y_val2, y_pred)
    # Otherwise, train the model from scratch
    elif upload == False:
        best_score = 0
        # Grid search to find the best hyperparameters for the Random Forest
        # 'bootstrap': Whether bootstrap samples are used when building trees
        for bootstrap in [True, False]:
            # 'max_features': The number of features to consider when looking for the best split
            for max_features in ['auto', 'sqrt']:
                # 'max_depth': Maximum depth of the tree
                for max_depth in Max_depth:
                    # 'min_samples_leaf': Minimum number of samples required at the leaf node
                    for min_samples_leaf in Min_samples_leaf:
                        # 'min_samples_split': Minimum number of samples required to split an internal node
                        for min_samples_split in Min_samples_split:
                                rf = RandomForestClassifier(bootstrap=bootstrap, max_features=max_features, max_depth=max_depth,
                                                          min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,
                                                          n_estimators = int(np.sqrt(len(X_train.columns))),random_state= 42)
                                rf.fit(X_train, y_train)
                                score = rf.score(X_val1, y_val1)
                                # Update best parameters if current score is better
                                if score > best_score:
                                    best_score = score
                                    best_parameters = {'bootstrap': bootstrap, 'max_features': max_features, 'max_depth': max_depth,
                                                      'min_samples_leaf': min_samples_leaf, 'min_samples_split': min_samples_split,
                                                      }
        max_features1 = best_parameters['max_features']
        bootstrap1 = best_parameters['bootstrap']
        max_depth1 = best_parameters['max_depth']
        bootstrap1 = best_parameters['bootstrap']
        min_samples_leaf1 = best_parameters['min_samples_leaf']
        min_samples_split1 = best_parameters['min_samples_split']
        # Training the model with the best parameters
        rf1 = RandomForestClassifier(bootstrap=bootstrap1, max_features=max_features1, max_depth=max_depth1,
                                                          min_samples_leaf=min_samples_leaf1, min_samples_split=min_samples_split1,
                                                          n_estimators = int(np.sqrt(len(X_train.columns))))
        rf1.fit(X_train, y_train)
        # Saving the model
        path1 = working_path + '/model_weights/' + ticker + 'rf.h5'
        os.makedirs(os.path.dirname(path1), exist_ok=True)
        dump(rf1, path1)

        y_pred = rf1.predict(X_val2)
        # Calculating the accuracy score
        test_score = rf1.score(X_val2,y_val2)
        # Calculating the balanced accuracy score
        balanced_score1 = balanced_accuracy_score(y_val2, y_pred)

    return test_score, balanced_score1


# # 6.XGBoost

# In[17]:


def XGBoost(ticker = 'USDEUR', feature_dic = None, upload = False, path = '/FX-Data/',
                         train_start = '2005-01-03', train_end = '2011-12-30',
                         vali1_start = '2012-01-02', vali1_end = '2014-12-31',
                         vali2_start = '2015-01-01', vali2_end = '2017-12-29',
                         test_start = '2018-01-01', test_end = '2018-12-31',
                         standarlization = True, from_beginning = True, Max_depth = [3,5,7,9],
                         Min_child_weight = [1, 2, 4], Gamma = [0, 0.1, 0.2], Subsample = [0.6, 0.8, 1],
                         Colsample_bytree = [0.6, 0.8, 1], Learning_rate = [0.001, 0.01, 0.1]):

    """
    Trains and validates an XGBoost model for a given ticker.

    Parameters:
    - ticker (str): Ticker symbol, default is 'USDEUR'.
    - feature_dic (dict or None): Dictionary of features for the given ticker. If None, no feature selection is done.
    - upload (bool): Whether to load a pre-trained model or not.
    - path (str): Path to data, default is '/FX-Data/'.
    - train_start (str): Start date for the training dataset.
    - train_end (str): End date for the training dataset.
    - vali1_start (str): Start date for the first validation dataset.
    - vali1_end (str): End date for the first validation dataset.
    - vali2_start (str): Start date for the second validation dataset.
    - vali2_end (str): End date for the second validation dataset.
    - test_start (str): Start date for the testing dataset.
    - test_end (str): End date for the testing dataset.
    - standarlization (bool): Whether to standardize data or not.
    - from_beginning (bool): If true, use data from the beginning.
    - Max_depth (list): List of the values of the Maximum depth of the tree that will be considered in the model.
    - Min_child_weight (list): List of the values of the Minimum sum of instance weight needed in a child that will be considered in the model.
    - Gamma (list): List of the values of the Regularization parameter that will be considered in the model.
    - Subsample (list): List of the values of the Proportion of training data to grow trees that will be considered in the model.
    - Colsample_bytree (list): List of the values of the Subsample ratio of columns when constructing each tree that will be considered in the model.
    - Learning_rate (list): List of the values of the step size at each iteration while moving towards a minimum of the loss function that will be considered in the model.

    Returns:
    - test_score (float): Accuracy score on the test data.
    - balanced_score1 (float): Balanced accuracy score on the test data.
    """

    if feature_dic is not None and not isinstance(feature_dic, dict):
        raise ValueError("feature_dic must be of type dict or None.")

    # Data split
    X_train, X_val1, X_val2, X_test, y_train, y_val1, y_val2, y_test = dp.getProcessedData(ticker, path,
                                                                             train_start, train_end,
                                                                             vali1_start, vali1_end,
                                                                             vali2_start, vali2_end,
                                                                             test_start, test_end,
                                                                             standarlization, from_beginning)
    working_path = os.getcwd()
    if feature_dic != None:
        # For each market, we only use several top features selected by SFFS.
        X_train = X_train.iloc[:,feature_dic[ticker]]
        X_val1 = X_val1.iloc[:,feature_dic[ticker]]
        X_val2 = X_val2.iloc[:,feature_dic[ticker]]
        X_test = X_test.iloc[:,feature_dic[ticker]]

    # Check if the model should be loaded directly
    if upload == True:
        # For ensure that the model does not change when it is used again and is easy to reproduce.
        xgb_model = load(working_path + '/model_weights/' + ticker +'xgb.h5')
        y_pred = xgb_model.predict(X_val2)
        # Calculating the accuracy score
        test_score = xgb_model.score(X_val2,y_val2)
        # Calculating the balanced accuracy score
        balanced_score1 = balanced_accuracy_score(y_val2, y_pred)
    elif upload == False:
        best_score = 0
        # Grid search to find the best hyperparameters for the XGBoost
        # 'max_depth': Maximum depth of a tree
        for max_depth in Max_depth:
            # 'min_child_weight': Minimum sum of instance weight needed in a child
            for min_child_weight in Min_child_weight:
                # 'gamma': Regularization parameter
                for gamma in Gamma:
                    # 'subsample': Proportion of training data to grow trees and prevent overfitting
                    for subsample in Subsample:
                        # 'colsample_bytree': Subsample ratio of columns when constructing each tree
                        for colsample_bytree in Colsample_bytree:
                            # 'learning_rate': Shrinks the feature weights to make the boosting process more conservative
                            for learning_rate in Learning_rate:
                                xgb_model = xgb.XGBClassifier(learning_rate=learning_rate, max_depth=max_depth,
                                                              min_child_weight=min_child_weight, gamma=gamma,
                                                              subsample=subsample, colsample_bytree=colsample_bytree,
                                                              use_label_encoder=False, eval_metric='logloss',
                                                              n_estimators = int(np.sqrt(len(X_train.columns))))
                                xgb_model.fit(X_train, y_train)
                                score = xgb_model.score(X_val1, y_val1)
                                # Update best parameters if current score is better
                                if score > best_score:
                                    best_score = score
                                    best_parameters = {'max_depth': max_depth, 'min_child_weight': min_child_weight,
                                                      'gamma': gamma, 'subsample': subsample, 'colsample_bytree': colsample_bytree,
                                                      'learning_rate': learning_rate}
        # Training the model with the best parameters
        xgb_model = xgb.XGBClassifier(learning_rate=best_parameters['learning_rate'], max_depth=best_parameters['max_depth'],
                                      min_child_weight=best_parameters['min_child_weight'], gamma=best_parameters['gamma'],
                                      subsample=best_parameters['subsample'], colsample_bytree=best_parameters['colsample_bytree'],
                                      use_label_encoder=False, eval_metric='logloss',
                                      n_estimators = int(np.sqrt(len(X_train.columns))))
        xgb_model.fit(X_train, y_train)
        # Saving the model
        path1 = working_path + '/model_weights/' + ticker +'xgb.h5'
        os.makedirs(os.path.dirname(path1), exist_ok=True)
        dump(xgb_model, path1)

        y_pred = xgb_model.predict(X_val2)
        # Calculating the accuracy score
        test_score = xgb_model.score(X_val2,y_val2)
        # Calculating the balanced accuracy score
        balanced_score1 = balanced_accuracy_score(y_val2, y_pred)

    return test_score, balanced_score1


# # LSTM

# In[10]:


def LSTM_model_build(hp, input_shape=20):

    """
    Build an LSTM model with specified hyperparameters and input shape.

    Parameters:
    - hp: Keras tuner hyperparameters object.
    - input_shape (int): shape of the input data, default is 20.

    Returns:
    - model (tf.keras.models.Model): Compiled LSTM model.
    """

    # Define the input shape.
    input = tf.keras.Input(shape=(input_shape,1))
    x = input

    # Dynamically add LSTM layers based on hyperparameters.
    # The loop will iterate based on the number of LSTM layers specified in hyperparameters.
    for i in range(hp.Int('num_lstm_layers', 1, 3, default = 2)):
        x = layers.LSTM(# 'units' determines the dimensionality of the output space
                        units = hp.Int('units_'+str(i), min_value = 32, max_value = 256, step = 32),
                        # 'activation' is the activation function to use
                        activation = hp.Choice('activation_'+str(i), values = ['tanh', 'relu']),
                        # 'return_sequences' is True for all but the last LSTM layer to ensure each LSTM returns sequences to the next one
                        return_sequences = True if i < hp.Int('num_lstm_layers', 1, 3, default = 2)-1 else False)(x)

        # Add a dropout layer after the first LSTM layer to prevent overfitting
        # Dropout randomly sets a fraction of the input units to 0 at each update during training, which helps to prevent overfitting
        if i == 0:
            x = layers.Dropout(rate=hp.Float('dropout_rate_'+str(i), 0.3, 0.6, step=0.1))(x)

    # Add output layer
    # Dense layer with a sigmoid activation function is used for binary classification tasks
    x = layers.Dense(1, activation = 'sigmoid')(x)

    model = tf.keras.models.Model(inputs=input, outputs=x)

    # Compile the model with optimizer, loss, and metrics
    model.compile(
        # The optimizer is Adam, which is a popular choice for training deep learning models
        optimizer=tf.keras.optimizers.Adam(
            # Learning rate hyperparameter with a logarithmic scale
            hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling='log')
        ),
        # 'binary_crossentropy' is used as the loss function for binary classification tasks
        loss = 'binary_crossentropy',
        # Measure the accuracy of the model during training and testing
        metrics = ['accuracy']
    )

    return model


# In[ ]:


def LSTM_fit(X_train_LSTM, X_val_LSTM, y_train_LSTM, y_val_LSTM, name):

    """
    Fit the LSTM model to the training data and validate it using validation data.

    Parameters:
    - X_train_LSTM (pd.DataFrame): Training input data.
    - X_val_LSTM (pd.DataFrame): Validation input data.
    - y_train_LSTM (pd.DataFrame): Training true label.
    - y_val_LSTM (pd.DataFrame): Validation true label.
    - name (str): Project name for the hyperband tuner.

    Returns:
    - best_model (tf.keras.models.Model): LSTM model with the best hyperparameters found during tuning.
    """

    # Construct a model using the given hyperparameters and input shape derived from the training data.
    def build_model(hp):
        return LSTM_model_build(hp, input_shape=X_train_LSTM.shape[1])

    # Callback to reduce learning rate when 'val_loss' (validation loss) has stopped improving
    # Reducing learning rate can help the model converge to a local minimum more precisely
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-4)
    working_path1 = os.getcwd()

    # Setting up the Hyperband tuner
    # Hyperband is an optimized version of random search which uses early stopping to speed up hyperparameter tuning
    tuner = Hyperband(
        # Model-building function defined previously
        build_model,
        # Optimization objective
        objective = 'val_accuracy',
        # Maximum number of epochs to train a single model
        max_epochs = 50,
        # Directory to save tuning logs and trained models
        directory = working_path1 + '/LSTM_for_FX',
        # Name of the project, helps in organizing different tuning sessions
        project_name = name,
        # Random seed for reproducibility
        seed = 42
    )

    # Early stopping callback to stop training when 'val_accuracy' (validation accuracy) stops improving
    # Helps prevent overfitting and reduce unnecessary training epochs
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience = 5)
    # Perform hyperparameter search
    history = tuner.search(
        # Training features
        X_train_LSTM,
        # Training labels
        y_train_LSTM,
        # Maximum number of epochs per model
        epochs=50,
        # Validation data
        # We use pre-divided validation set due to LSTM is a model for processing time series type data
        validation_data = (X_val_LSTM, y_val_LSTM),
        # Not shuffling as it's time series data
        shuffle = False,
        # Number of samples per gradient update
        batch_size=32,
        # List of callbacks to apply during training
        callbacks=[stop_early,reduce_lr]
    )

    # Get the best hyperparameters and build the best model
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.hypermodel.build(best_hp)

    return best_model


# In[12]:


def LSTM_predict(best_model, X_test):

    """
    Predict the output using the provided LSTM model and test data.

    Parameters:
    - best_model (tf.keras.models.Model): Trained LSTM model.
    - X_test (pd.DataFrame): Test input data.

    Returns:
    - pred (list): Binary predictions.
    - test_predict (np.array): Raw model outputs.
    """

    test_predict = best_model.predict(X_test)

    pred = []

    for j in test_predict:
        # Convert probabilities into class labels using 0.5 as threshold
        if(j>0.5):
            pred.append(1)
        else:
            pred.append(0)

    return pred, test_predict


# In[ ]:


def LSTM(ticker = 'USDEUR', feature_dic = None, upload = False, path = '/FX-Data/',
                         train_start = '2005-01-03', train_end = '2011-12-30',
                         vali1_start = '2012-01-02', vali1_end = '2014-12-31',
                         vali2_start = '2015-01-01', vali2_end = '2017-12-29',
                         test_start = '2018-01-01', test_end = '2018-12-31',
                         standarlization = True, from_beginning = True):

    """
    Full LSTM model process, from data loading to prediction.

    Parameters:
    - ticker (str): Ticker symbol, default is 'USDEUR'.
    - feature_dic (dict or None): Dictionary of features for the given ticker. If None, no feature selection is done.
    - upload (bool): Whether to load a pre-trained model or not.
    - path (str): Path to data, default is '/FX-Data/'.
    - train_start (str): Start date for the training dataset.
    - train_end (str): End date for the training dataset.
    - vali1_start (str): Start date for the first validation dataset.
    - vali1_end (str): End date for the first validation dataset.
    - vali2_start (str): Start date for the second validation dataset.
    - vali2_end (str): End date for the second validation dataset.
    - test_start (str): Start date for the testing dataset.
    - test_end (str): End date for the testing dataset.
    - standarlization (bool): Whether to standardize data or not.
    - from_beginning (bool): If true, use data from the beginning.
    
    Returns:
    - test_score (float): Accuracy score on the test data.
    - balanced_score1 (float): Balanced accuracy score on the test data.
    """

    if feature_dic is not None and not isinstance(feature_dic, dict):
        raise ValueError("feature_dic must be of type dict or None.")

    # Data split
    X_train, X_val1, X_val2, X_test, y_train, y_val1, y_val2, y_test = dp.getProcessedData(ticker, path,
                                                                             train_start, train_end,
                                                                             vali1_start, vali1_end,
                                                                             vali2_start, vali2_end,
                                                                             test_start, test_end,
                                                                             standarlization, from_beginning)
    working_path = os.getcwd()
    if feature_dic != None:
        # For each market, we only use several top features selected by SFFS.
        X_train = X_train.iloc[:,feature_dic[ticker]]
        X_val1 = X_val1.iloc[:,feature_dic[ticker]]
        X_val2 = X_val2.iloc[:,feature_dic[ticker]]
        X_test = X_test.iloc[:,feature_dic[ticker]]
        name = 'history_' + ticker

    # Check if the model should be loaded directly
    if upload == True:
        # For ensure that the model does not change when it is used again and is easy to reproduce.
        best_model = tf.keras.models.load_model(working_path + '/LSTM_for_FX/LSTM_' + ticker + '.h5')
    # Otherwise, train the model from scratch
    elif upload == False:
        # Obtain the best model
        best_model = LSTM_fit(X_train, X_val1, y_train, y_val1, name)
        # Save as a h5 format file            
        path1 = working_path + '/LSTM_for_FX/LSTM_' + ticker + '.h5'
        os.makedirs(os.path.dirname(path1), exist_ok=True)
        best_model.save(path1)

    # Visualize model structure
    tf.keras.utils.plot_model(
        best_model,
        to_file = working_path + '/LSTM_for_FX/LSTM_' + ticker + '.png',
    )

    # Predict the test set results
    y_pred, test_predict = LSTM_predict(best_model, X_val2)

    # accuracy scores
    test_score = accuracy_score(y_val2,y_pred)
    # balanced accuracy scores
    balanced_score1 = balanced_accuracy_score(y_val2,y_pred)

    return test_score, balanced_score1


# # Statistical Learning Result Comparison

# In[18]:


def SL_Compare(ticker_names = ['USDEUR', 'USDJPY', 'USDGBP', 'USDCHF', 'USDNZD', 'USDCAD', 'USDSEK', 'USDDKK', 'USDNOK',
                                  'EURJPY', 'EURGBP', 'EURCHF', 'EURNZD', 'EURCAD', 'EURSEK', 'EURDKK', 'EURNOK'],
                         feature_dic = None, upload = False, path = '/FX-Data/',
                         train_start = '2005-01-03', train_end = '2011-12-30',
                         vali1_start = '2012-01-02', vali1_end = '2014-12-31',
                         vali2_start = '2015-01-01', vali2_end = '2017-12-29',
                         test_start = '2018-01-01', test_end = '2018-12-31',
                         standarlization = True, from_beginning = True, RF_max_depth = [3,5,7,9],
                         RF_min_samples_leaf = [1, 2, 4], RF_min_samples_split = [2, 5, 10], XGB_max_depth = [3,5,7,9],
                         XGB_min_child_weight = [1, 2, 4], XGB_gamma = [0, 0.1, 0.2], XGB_subsample = [0.6, 0.8, 1],
                         XGB_colsample_bytree = [0.6, 0.8, 1], XGB_learning_rate = [0.001, 0.01, 0.1]):

    """
    Compare performance of various statistical learning methods.

    Parameters:
    - ticker_names (list): List of currency ticker names.
    - feature_dic (dict or None): Dictionary of features for the given ticker. If None, no feature selection is done.
    - upload (bool): Whether to load a pre-trained model or not.
    - path (str): Path to data, default is '/FX-Data/'.
    - train_start (str): Start date for the training dataset.
    - train_end (str): End date for the training dataset.
    - vali1_start (str): Start date for the first validation dataset.
    - vali1_end (str): End date for the first validation dataset.
    - vali2_start (str): Start date for the second validation dataset.
    - vali2_end (str): End date for the second validation dataset.
    - test_start (str): Start date for the testing dataset.
    - test_end (str): End date for the testing dataset.
    - standarlization (bool): Whether to standardize data or not.
    - from_beginning (bool): If true, use data from the beginning.
    - RF_max_depth (list): List of the values of the Maximum depth of the tree that will be considered in the Random Forest model.
    - RF_min_samples_leaf (list): List of the values of the Minimum number of samples required at the leaf node that will be considered in the Random Forest model.
    - RF_min_samples_split (list): List of the values of the Minimum number of samples required to split an internal node that will be considered in the Random Forest model.
    - XGB_max_depth (list): List of the values of the Maximum depth of the tree that will be considered in the XGBoost model.
    - XGB_min_child_weight (list): List of the values of the Minimum sum of instance weight needed in a child that will be considered in the XGBoost model.
    - XGB_gamma (list): List of the values of the Regularization parameter that will be considered in the XGBoost model.
    - XGB_subsample (list): List of the values of the Proportion of training data to grow trees that will be considered in the XGBoost model.
    - XGB_colsample_bytree (list): List of the values of the Subsample ratio of columns when constructing each tree that will be considered in the XGBoost model.
    - XGB_learning_rate (list): List of the values of the step size at each iteration while moving towards a minimum of the loss function that will be considered in the XGBoost model.

    Returns:
    - acc_df (pd.DataFrame): Dataframe containing accuracy of various models.
    """

    if feature_dic is not None and not isinstance(feature_dic, dict):
        raise ValueError("feature_dic must be of type dict or None.")

    # Lists to store accuracy scores and balanced accuracy scores for each model and market
    Stnm = ticker_names
    svmnon=[]
    lr=[]
    lda = []
    qda = []
    rf=[]
    xgboost=[]
    lstm = []

    svmnon_balance=[]
    lr_balance=[]
    lda_balance = []
    qda_balance = []
    rf_balance = []
    xgboost_balance = []
    lstm_balance = []

    # Loop through each market
    for ticker in ticker_names:
        a1,b1 = SVM_Gaussian(ticker = ticker, feature_dic = feature_dic, upload = upload, path = path,
                         train_start = train_start, train_end = train_end,
                         vali1_start = vali1_start, vali1_end = vali1_end,
                         vali2_start = vali2_start, vali2_end = vali2_end,
                         test_start = test_start, test_end = test_end,
                         standarlization = standarlization, from_beginning = from_beginning)
        a2,b2 = LR_Lasso(ticker = ticker, feature_dic = feature_dic, upload = upload, path = path,
                         train_start = train_start, train_end = train_end,
                         vali1_start = vali1_start, vali1_end = vali1_end,
                         vali2_start = vali2_start, vali2_end = vali2_end,
                         test_start = test_start, test_end = test_end,
                         standarlization = standarlization, from_beginning = from_beginning)
        a3,b3 = LDA(ticker = ticker, feature_dic = feature_dic, upload = upload, path = path,
                         train_start = train_start, train_end = train_end,
                         vali1_start = vali1_start, vali1_end = vali1_end,
                         vali2_start = vali2_start, vali2_end = vali2_end,
                         test_start = test_start, test_end = test_end,
                         standarlization = standarlization, from_beginning = from_beginning)
        a4,b4 = QDA(ticker = ticker, feature_dic = feature_dic, upload = upload, path = path,
                         train_start = train_start, train_end = train_end,
                         vali1_start = vali1_start, vali1_end = vali1_end,
                         vali2_start = vali2_start, vali2_end = vali2_end,
                         test_start = test_start, test_end = test_end,
                         standarlization = standarlization, from_beginning = from_beginning)
        a5,b5 = Random_Forest(ticker = ticker, feature_dic = feature_dic, upload = upload, path = path,
                         train_start = train_start, train_end = train_end,
                         vali1_start = vali1_start, vali1_end = vali1_end,
                         vali2_start = vali2_start, vali2_end = vali2_end,
                         test_start = test_start, test_end = test_end,
                         standarlization = standarlization, from_beginning = from_beginning,
                         Max_depth = RF_max_depth, Min_samples_leaf = RF_min_samples_leaf,
                         Min_samples_split = RF_min_samples_split)
        a6,b6 = XGBoost(ticker = ticker, feature_dic = feature_dic, upload = upload, path = path,
                         train_start = train_start, train_end = train_end,
                         vali1_start = vali1_start, vali1_end = vali1_end,
                         vali2_start = vali2_start, vali2_end = vali2_end,
                         test_start = test_start, test_end = test_end,
                         standarlization = standarlization, from_beginning = from_beginning,
                         Max_depth = XGB_max_depth, Min_child_weight = XGB_min_child_weight,
                         Gamma = XGB_gamma, Subsample = XGB_subsample,
                         Colsample_bytree = XGB_colsample_bytree, Learning_rate = XGB_learning_rate)
        a7,b7 = LSTM(ticker = ticker, feature_dic = feature_dic, upload = upload, path = path,
                         train_start = train_start, train_end = train_end,
                         vali1_start = vali1_start, vali1_end = vali1_end,
                         vali2_start = vali2_start, vali2_end = vali2_end,
                         test_start = test_start, test_end = test_end,
                         standarlization = standarlization, from_beginning = from_beginning)

        # Append scores to the lists
        svmnon.append(a1)
        lr.append(a2)
        lda.append(a3)
        qda.append(a4)
        rf.append(a5)
        xgboost.append(a6)
        lstm.append(a7)

        svmnon_balance.append(b1)
        lr_balance.append(b2)
        lda_balance.append(b3)
        qda_balance.append(b4)
        rf_balance.append(b5)
        xgboost_balance.append(b6)
        lstm_balance.append(b7)


    # Create a dataframe to compile all the accuracy scores
    acc_df=pd.DataFrame(zip(Stnm,svmnon, svmnon_balance, lr, lr_balance, lda, lda_balance, qda, qda_balance, rf, rf_balance, xgboost, xgboost_balance, lstm, lstm_balance),columns=['Stock','SVM Gaussian Kernel','SVM Gaussian Kernel balance',
                                                         'Logistic Regression','Logistic Regression balance','LDA','LDA balance','QDA','QDA balance','Random Forest','Random Forest balance','XGBoost','XGBoost balance', 'LSTM', 'LSTM balance'])
    working_path = os.getcwd()
    acc_df.to_csv(working_path + '/Statistical_Learning_Result_Comparison.csv', index=False)
    return acc_df

