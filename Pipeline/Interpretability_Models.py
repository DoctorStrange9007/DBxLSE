#!/usr/bin/env python
# coding: utf-8

# In[19]:


import subprocess
libraries_to_install = ['lime', 'openpyxl']

for library in libraries_to_install:
    subprocess.run(["pip", "install", library], check=True)


# In[37]:


import Data_Processing as dp
import Machine_Learning_Models as ml
import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import math
import multiprocessing
import matplotlib.pyplot as plt
from datetime import date
import glob, os
from joblib import dump, load
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, balanced_accuracy_score
import time
import lime
from lime import submodular_pick
from lime import lime_tabular
from collections import defaultdict
import openpyxl


# # XGB Feature Importance (Gain)

# In[7]:


def XGB_Feature_Importance(ticker_names = ['USDEUR', 'USDJPY', 'USDGBP', 'USDCHF', 'USDNZD', 'USDCAD', 'USDSEK', 'USDDKK', 'USDNOK',
                                  'EURJPY', 'EURGBP', 'EURCHF', 'EURNZD', 'EURCAD', 'EURSEK', 'EURDKK', 'EURNOK'], feature_dic = None, path = '/FX-Data/',
                         train_start = '2005-01-03', train_end = '2011-12-30',
                         vali1_start = '2012-01-02', vali1_end = '2014-12-31',
                         vali2_start = '2015-01-01', vali2_end = '2017-12-29',
                         test_start = '2018-01-01', test_end = '2018-12-31',
                         standarlization = True, from_beginning = True, top_n = 5):

    """
    Compute feature importance using XGBoost Built-in Feature Importance for a list of tickers.
    
    Parameters:
    - ticker_names (list): List of currency ticker names.
    - feature_dic (dict or None): Dictionary of features for the given ticker. If None, no feature selection is done.
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
    - top_n (int): The number of important features to filter out, default is 5.
    
    Returns:
    - feature_importance (dict): Dictionary with tickers as keys and their respective feature importances as values.
    - xgbfi_gain_top_n (dict): Dictionary with tickers as keys and top n important features based on their respective feature importances as values.
    """

    if feature_dic is not None and not isinstance(feature_dic, dict):
        raise ValueError("feature_dic must be of type dict or None.")

    working_path = os.getcwd()
    # List to store the XGB feature importance for each ticker
    xgbfi_gain_result = []
    # Dictionary to store the top n important features for each ticker
    xgbfi_gain_top_n = {}

    # Loop through each ticker
    for i in range(len(ticker_names)):
        ticker = ticker_names[i]
        # Data split
        X_train, X_val1, X_val2, X_test, y_train, y_val1, y_val2, y_test = dp.getProcessedData(ticker, path,
                                                                                     train_start, train_end,
                                                                                     vali1_start, vali1_end,
                                                                                     vali2_start, vali2_end,
                                                                                     test_start, test_end,
                                                                                     standarlization, from_beginning)
        if feature_dic != None:
            # For each market, we only use several top features selected by SFFS.
            X_train = X_train.iloc[:,feature_dic[ticker]]
            X_val1 = X_val1.iloc[:,feature_dic[ticker]]
            X_val2 = X_val2.iloc[:,feature_dic[ticker]]
            X_test = X_test.iloc[:,feature_dic[ticker]]

        # Merge the training sets and validation sets to form a complete training set
        X_train_total = pd.concat([X_train, X_val1, X_val2])
        y_train_total = pd.concat([y_train, y_val1, y_val2])

        # Load the XGBoost model with the previously found best hyperparameters
        xgb_model = load(working_path + '/model_weights/' + ticker +'xgb.h5')
        # Retrain the XGBoost model parameters on the newly defined training set
        xgb_model.fit(X_train_total, y_train_total)

        # Extract XGB feature importance from the trained XGBoost model 
        # Calculated by the average gain across all splits the feature is used in
        result = xgb_model.get_booster().get_score(importance_type='gain')

        # Sort the results by feature importance
        ord = dict(sorted(result.items(), key=lambda item: item[1],reverse=True))

        # Append the sorted feature importance results to the list
        xgbfi_gain_result.append(ord)

        # Top n important features selected by XGB feature importance
        lst = list(ord)

        feat_n = lst[:top_n]

        xgbfi_gain_top_n[ticker] = feat_n

    path1 = working_path + '/XGB_FI_result/xgbfi_gain_result.h5'
    path2 = working_path + '/XGB_FI_result/xgbfi_gain_top_n.h5'
    os.makedirs(os.path.dirname(path1), exist_ok=True)
    os.makedirs(os.path.dirname(path2), exist_ok=True)
    # Save the complete feature importance results to a h5 file
    dump(xgbfi_gain_result, path1)
    # Save the top n features for each ticker to a h5 file
    dump(xgbfi_gain_top_n, path2)

    return xgbfi_gain_result, xgbfi_gain_top_n


# # PFI

# In[16]:


def PFI(ticker_names = ['USDEUR', 'USDJPY', 'USDGBP', 'USDCHF', 'USDNZD', 'USDCAD', 'USDSEK', 'USDDKK', 'USDNOK',
                                  'EURJPY', 'EURGBP', 'EURCHF', 'EURNZD', 'EURCAD', 'EURSEK', 'EURDKK', 'EURNOK'], feature_dic = None, path = '/FX-Data/',
                         train_start = '2005-01-03', train_end = '2011-12-30',
                         vali1_start = '2012-01-02', vali1_end = '2014-12-31',
                         vali2_start = '2015-01-01', vali2_end = '2017-12-29',
                         test_start = '2018-01-01', test_end = '2018-12-31',
                         standarlization = True, from_beginning = True,
                         model = 'xgb', visualization = True, top_n = 5):

    """
    Evaluate feature importance using the Permutation Feature Importance (PFI) method for a list of tickers.
    
    Parameters:
    - ticker_names (list): List of currency ticker names.
    - feature_dic (dict or None): Dictionary of features for the given ticker. If None, no feature selection is done.
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
    - model (str): Model to be interpreted. Options include:
        - 'svm': Support Vector Machine
        - 'lr': Logistic Regression
        - 'lda': Linear Discriminant Analysis
        - 'qda': Quadratic Discriminant Analysis
        - 'rf': Random Forest
        - 'xgb': XGBoost
        - 'lstm': Long Short Term Memory neural network
    - visualization (bool): If true, visualize the result.
    - top_n (int): The number of important features to filter out, default is 5.
    
    Returns:
    - pfi_importance (dict): Dictionary with tickers as keys and their respective PFI scores as values.
    - pfi_top_n (dict): Dictionary with tickers as keys and top n important features based on their respective PFI scores as values.
    """

    if feature_dic is not None and not isinstance(feature_dic, dict):
        raise ValueError("feature_dic must be of type dict or None.")

    # List to store the result of permutation feature importance for each ticker
    pfi_result = []
    # Dictionary to store features for each ticker
    stock_columns = {}

    working_path = os.getcwd()
    # Get the number of available CPU cores for parallel processing
    n_cores = multiprocessing.cpu_count()

    # Loop through each ticker
    for i in range(len(ticker_names)):
        ticker = ticker_names[i]
        # Data split
        X_train, X_val1, X_val2, X_test, y_train, y_val1, y_val2, y_test = dp.getProcessedData(ticker, path,
                                                                                     train_start, train_end,
                                                                                     vali1_start, vali1_end,
                                                                                     vali2_start, vali2_end,
                                                                                     test_start, test_end,
                                                                                     standarlization, from_beginning)
        if feature_dic != None:
            # For each market, we only use several top features selected by SFFS.
            X_train = X_train.iloc[:,feature_dic[ticker]]
            X_val1 = X_val1.iloc[:,feature_dic[ticker]]
            X_val2 = X_val2.iloc[:,feature_dic[ticker]]
            X_test = X_test.iloc[:,feature_dic[ticker]]

        if model == 'svm':
            p1 = working_path + '/model_weights/' + ticker +'svmnon.h5'
        elif model == 'lr':
            p1 = working_path + '/model_weights/' + ticker + 'lr.h5'
        elif model == 'lda':
            p1 = working_path + '/model_weights/' + ticker + 'lda.h5'
        elif model == 'qda':
            p1 = working_path + '/model_weights/' + ticker + 'qda.h5'
        elif model == 'rf':
            p1 = working_path + '/model_weights/' + ticker + 'rf.h5'
        elif model == 'xgb':
            p1 = working_path + '/model_weights/' + ticker +'xgb.h5'

        if model == 'lstm':
            # Use previous best hyper parameters obtained from model trained on train and use val1 as validation set
            base_model = tf.keras.models.load_model(working_path + '/LSTM_for_FX/LSTM_' + ticker + '.h5')
        else:
            # Use previous best hyper parameters obtained from model trained on train and use val1 as validation set
            base_model = load(p1)

        # Store column names for the current stock
        stock_columns[ticker] = X_train.columns

        # Compute the permutation importance  on the val2 set
        result = permutation_importance(base_model, X_val2, y_val2,
                                n_jobs=n_cores, n_repeats=5, random_state=42)

        # Append the permutation importance result
        pfi_result.append(result)

    # List to store the top n important features for each ticker
    pfi_top_n = {}
    j = 0
    for key in stock_columns.keys():
        # Get indices of top n features for current ticker
        feat_n = np.argsort(pfi_result[j]['importances_mean'])[::-1][:, top_n]
        # Convert feature indices to feature names
        ls = [stock_columns[key][i] for i in feat_n]
        pfi_top_n[key] = ls
        j+=1

    path1 = working_path + '/PFI_result/PFI_result.h5'
    path2 = working_path + '/PFI_result/PFI_top_n.h5'
    os.makedirs(os.path.dirname(path1), exist_ok=True)
    os.makedirs(os.path.dirname(path2), exist_ok=True)
    # Save the complete feature importance results to a h5 file
    dump(pfi_result, path1)
    # Save the top n features for each ticker to a h5 file
    dump(pfi_top_n, path2)

    if visualization == True:
        # Dictionary to store mean importance for each feature of each ticker
        ticker_imp = {}
        for i in range(len(pfi_result)):
            ticker_imp[ticker_names[i]] = pfi_result[i]['importances_mean']
        # Visualize the importance of each feature for each ticker
        for i in range(len(pfi_result)):
            # Sort features by importance
            sorted_idx = ticker_imp[ticker_names[i]].argsort()
            # Plotting
            fig, ax = plt.subplots()
            ax.barh(stock_columns[ticker_names[i]][sorted_idx], pfi_result[i].importances[sorted_idx].mean(axis=1).T)
            ax.set_title("Permutation Importances (test set) "+ticker_names[i])
            fig.tight_layout()
            plt.show()
    return pfi_result, pfi_top_n


# # SP-LIME

# In[35]:


def SP_LIME(ticker_names = ['USDEUR', 'USDJPY', 'USDGBP', 'USDCHF', 'USDNZD', 'USDCAD', 'USDSEK', 'USDDKK', 'USDNOK',
                                  'EURJPY', 'EURGBP', 'EURCHF', 'EURNZD', 'EURCAD', 'EURSEK', 'EURDKK', 'EURNOK'], feature_dic = None, path = '/FX-Data/',
                         train_start = '2005-01-03', train_end = '2011-12-30',
                         vali1_start = '2012-01-02', vali1_end = '2014-12-31',
                         vali2_start = '2015-01-01', vali2_end = '2017-12-29',
                         test_start = '2018-01-01', test_end = '2018-12-31',
                         standarlization = True, from_beginning = True,
                         model = 'xgb', visualization = True, top_n = 5,
                         num_features = 20, num_exps_desired=156):

    """
    Apply the LIME (Local Interpretable Model-agnostic Explanations) method with Submodular Pick on a set of currency tickers. 

    Parameters:
    - ticker_names (list): List of currency ticker names.
    - feature_dic (dict or None): Dictionary of features for the given ticker. If None, no feature selection is done.
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
    - model (str): Model to be interpreted. Options include:
        - 'svm': Support Vector Machine
        - 'lr': Logistic Regression
        - 'lda': Linear Discriminant Analysis
        - 'qda': Quadratic Discriminant Analysis
        - 'rf': Random Forest
        - 'xgb': XGBoost
        - 'lstm': Long Short Term Memory neural network
    - visualization (bool): If true, visualize the result.
    - top_n (int): The number of important features to filter out, default is 5.
    - num_features (int): Number of features for the LIME explanations. Default is 20.
    - num_exps_desired (int): Number of explanations desired from the Submodular Pick. Default is 156.

    Returns:
    - sp_lime_result (dict): Dictionary with tickers as keys and their respective LIME explanations as values.
    - sp_lime_top_n (dict): Dictionary with tickers as keys and top n important features based on LIME explanations as values.
    """

    if feature_dic is not None and not isinstance(feature_dic, dict):
        raise ValueError("feature_dic must be of type dict or None.")

    # Dictionary to store LIME results for each ticker
    sp_lime_result = dict.fromkeys(ticker_names, None)
    working_path = os.getcwd()

    # Loop through each ticker
    for i in range(len(ticker_names)):
        ticker = ticker_names[i]
        # Data split
        X_train, X_val1, X_val2, X_test, y_train, y_val1, y_val2, y_test = dp.getProcessedData(ticker, path,
                                                                                     train_start, train_end,
                                                                                     vali1_start, vali1_end,
                                                                                     vali2_start, vali2_end,
                                                                                     test_start, test_end,
                                                                                     standarlization, from_beginning)
        if feature_dic != None:
            # For each market, we only use several top features selected by SFFS.
            X_train = X_train.iloc[:,feature_dic[ticker]]
            X_val1 = X_val1.iloc[:,feature_dic[ticker]]
            X_val2 = X_val2.iloc[:,feature_dic[ticker]]
            X_test = X_test.iloc[:,feature_dic[ticker]]

        if model == 'svm':
            p1 = working_path + '/model_weights/' + ticker +'svmnon.h5'
        elif model == 'lr':
            p1 = working_path + '/model_weights/' + ticker + 'lr.h5'
        elif model == 'lda':
            p1 = working_path + '/model_weights/' + ticker + 'lda.h5'
        elif model == 'qda':
            p1 = working_path + '/model_weights/' + ticker + 'qda.h5'
        elif model == 'rf':
            p1 = working_path + '/model_weights/' + ticker + 'rf.h5'
        elif model == 'xgb':
            p1 = working_path + '/model_weights/' + ticker +'xgb.h5'

        if model == 'lstm':
            # Use previous best hyper parameters obtained from model trained on train and use val1 as validation set
            base_model = tf.keras.models.load_model(working_path + '/LSTM_for_FX/LSTM_' + ticker + '.h5')
        else:
            # Use previous best hyper parameters obtained from model trained on train and use val1 as validation set
            base_model = load(p1)

        # Initialize a dictionary to store LIME results
        result_global_lime = dict.fromkeys(X_val2.columns, None)

        # Define a LIME explainer with train and validation data
        explainer = lime_tabular.LimeTabularExplainer(training_data = np.array(pd.concat([X_train, X_val1], axis = 0)),
                                                       feature_names = pd.concat([X_train, X_val1], axis = 0).columns,
                                                       training_labels = pd.concat([y_train, y_val1], axis = 0),
                                                       mode = 'classification', discretize_continuous = False, random_state= 42)

        # Define a function to output prediction probability
        def predictlime(x):
            if model == 'lstm':
                return base_model.predict(x)
            else:
                return base_model.predict_proba(x)

        # Define a submodular pick object
        sp_obj = submodular_pick.SubmodularPick(explainer, X_val2.values, predictlime,
                                        num_features = num_features,
                                        num_exps_desired = num_exps_desired)

        # Parse and store LIME explanations
        result_global_lime = dict.fromkeys(X_val2.columns, [])
        for ele in sp_obj.sp_explanations:
            if 1 in ele.as_map().keys():
                for key in dict(ele.as_list()):
                    result_global_lime[key] = result_global_lime[key] + [dict(ele.as_list())[key]]
            else:
                for key in dict(ele.as_list(label=0)):
                    result_global_lime[key] = result_global_lime[key] + [dict(ele.as_list(label=0))[key]]

        sp_lime_result[ticker_names[i]] = result_global_lime

    # Obtain top n features for each market
    final_importance_mkts = {}
    averages = {}
    sp_lime_top_n = {}
    for mkt in sp_lime_result.keys():
        for var in sp_lime_result[mkt].keys():
            averages[var] = np.mean([np.abs(x) for x in sp_lime_result[mkt][var]])
        sorted_important_variables = sorted(averages.items(), key=lambda x:x[1])
        sorted_important_variables= sorted_important_variables[len(sorted_important_variables)-top_n:]
        final_importance_mkts[mkt] = sorted_important_variables

    for mkt in final_importance_mkts.keys():
        for tpl in final_importance_mkts[mkt]:
            if sp_lime_top_n.get(mkt) is None:
                sp_lime_top_n[mkt] = [tpl[0]]
            else:
                sp_lime_top_n[mkt].append(tpl[0])

    path1 = working_path + '/SP_LIME_result_new/SP_LIME_result.h5'
    path2 = working_path + '/SP_LIME_result_new/SP_LIME_Top_n.h5'
    os.makedirs(os.path.dirname(path1), exist_ok=True)
    os.makedirs(os.path.dirname(path2), exist_ok=True)
    # Save the complete feature importance results to a h5 file
    dump(sp_lime_result, path1)
    # Save the top n features for each ticker to a h5 file
    dump(sp_lime_top_n, path2)

    if visualization == True:
        # model visualization
        for mkt in sp_lime_result.keys():
            num_features = len(sp_lime_result[mkt].keys())
            num_rows = num_features // 10 + (1 if num_features % 10 else 0)
            fig, axes = plt.subplots(num_rows, 10, figsize=(20, 10 * num_rows), sharex=True)

            fig.suptitle(mkt, fontsize=16)

            if num_rows == 1:
                axes = [[axes]]
            else:
                axes = axes.tolist()

            for idx, key in enumerate(sp_lime_result[mkt].keys()):
                row_idx = idx // 10
                col_idx = idx % 10
                sns.kdeplot(sp_lime_result[mkt][key], ax=axes[row_idx][col_idx])
                axes[row_idx][col_idx].axvline(x=0, color='r', linestyle='-')
                axes[row_idx][col_idx].set_title(key)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

    return sp_lime_result, sp_lime_top_n


# # Random vs SP_LIME vs PFI (vs XGBoost Feature Importance)

# In[ ]:


def generate_table_for_type(df, type_name):

    """
    Generates a table for a specific type from a given dataframe and calculates ranks 
    for each evaluation metric (Accuracy, Balanced Accuracy, F1 Score).

    Parameters:
    - df (pd.DataFrame): Dataframe containing evaluation results.
    - type_name (str): Name of the type for which the table is to be generated.

    Returns:
    - sub_df (pd.DataFrame): Generated table for the specific type.
    """

    # Filter dataframe by type
    sub_df = df[df['Type'] == type_name].copy()

    # Calculate ranks for each evaluation metric
    for col in ['Accuracy', 'Balanced Accuracy', 'F1 Score']:
        sub_df[col + ' Rank'] = df.groupby('Market')[col].rank(ascending=False).loc[sub_df.index].apply(math.floor)

    # Select required columns and return
    sub_df = sub_df[['Type', 'Market', 'Accuracy Rank', 'Balanced Accuracy Rank', 'F1 Score Rank']]
    return sub_df


# In[ ]:


def Interpret_Compare(ticker_names = ['USDEUR', 'USDJPY', 'USDGBP', 'USDCHF', 'USDNZD', 'USDCAD', 'USDSEK', 'USDDKK', 'USDNOK',
                                  'EURJPY', 'EURGBP', 'EURCHF', 'EURNZD', 'EURCAD', 'EURSEK', 'EURDKK', 'EURNOK'], path = '/FX-Data/',
                         train_start = '2005-01-03', train_end = '2011-12-30',
                         vali1_start = '2012-01-02', vali1_end = '2014-12-31',
                         vali2_start = '2015-01-01', vali2_end = '2017-12-29',
                         test_start = '2018-01-01', test_end = '2018-12-31',
                         standarlization = True, from_beginning = True,
                         model = 'xgb', sp_lime_top_n = None, pfi_top_n = None,
                         xgbfi_gain_top_n = None, self_selected_feature_dic = None):

    """
    Compares across different interpretability methods (like LIME, PFI, XGB etc.).

    Parameters:
    - ticker_names (list): List of currency ticker names.
    - feature_dic (dict or None): Dictionary of features for the given ticker. If None, no feature selection is done.
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
    - model (str): Model to be interpreted. Options include:
        - 'svm': Support Vector Machine
        - 'lr': Logistic Regression
        - 'lda': Linear Discriminant Analysis
        - 'qda': Quadratic Discriminant Analysis
        - 'rf': Random Forest
        - 'xgb': XGBoost
        - 'lstm': Long Short Term Memory neural network
    - sp_lime_top_n (dict or None): Dictionary with tickers as keys and top n important features based on LIME explanations as values.
    - pfi_top_n (dict or None): Dictionary with tickers as keys and top n important features based on their respective PFI scores as values.
    - xgbfi_gain_top_n (dict or None): Dictionary with tickers as keys and top n important features based on their respective feature importances as values.
    - self_selected_feature_dic (dict or None): Dictionary with tickers as keys and n self selected features as values.

    Returns:
    - tuple: Comprising of overall result dataframe, dataframes for LIME, PFI, XGB, and self-selected features, and ranking tables for various methods.
    """

    if sp_lime_top_n is not None and not isinstance(sp_lime_top_n, dict):
        raise ValueError("sp_lime_top_n must be of type dict or None.")

    if pfi_top_n is not None and not isinstance(pfi_top_n, dict):
        raise ValueError("pfi_top_n must be of type dict or None.")

    if xgbfi_gain_top_n is not None and not isinstance(xgbfi_gain_top_n, dict):
        raise ValueError("xgbfi_gain_top_n must be of type dict or None.")

    if self_selected_feature_dic is not None and not isinstance(self_selected_feature_dic, dict):
        raise ValueError("self_selected_feature_dic must be of type dict or None.")

    # Initializing an DataFrame to store final results
    df_all = pd.DataFrame()
    # Lists to store evaluation metrics and types of feature importance methods
    accuracy_scores = []
    f1_scores = []
    balanced_accuracy = []
    type_ls = []

    working_path = os.getcwd()

    # Loop through each currency pair ticker
    for i in range(len(ticker_names)):
        ticker = ticker_names[i]
        # Data split
        X_train, X_val1, X_val2, X_test, y_train, y_val1, y_val2, y_test = dp.getProcessedData(ticker, path,
                                                                                     train_start, train_end,
                                                                                     vali1_start, vali1_end,
                                                                                     vali2_start, vali2_end,
                                                                                     test_start, test_end,
                                                                                     standarlization, from_beginning)
        # Ensure that XGBoost Feature Importance is not applied to other models
        if model != 'xgb':
            xgbfi_gain_top_n = None
        # Get the top n important features determined by different methods (LIME, PFI, RANDOM, (XGB)) and store the important features in a dictionary
        important_variable_sets = {}
        if sp_lime_top_n != None:
            important_LIME = sp_lime_top_n[ticker]
            important_variable_sets['LIME'] = important_LIME
        if pfi_top_n != None:
            important_PFI = pfi_top_n[ticker]
            important_variable_sets['PFI'] = important_PFI
        if xgbfi_gain_top_n != None:
            important_XGB = xgbfi_gain_top_n[ticker]
            important_variable_sets['XGB'] = important_XGB
        if self_selected_feature_dic != None:
            important_Self_Selected = self_selected_feature_dic[ticker]
            important_variable_sets['Self_Selected'] = important_Self_Selected

        # Loop through each feature importance type
        for type in important_variable_sets.keys():

            pred = []
            # Use all of train, val1 and val2 sets for train
            X_train_total = pd.concat([X_train, X_val1, X_val2])
            y_train_total = pd.concat([y_train, y_val1, y_val2])

            # Filter the train and test data for the important features
            X_train_total = X_train_total.loc[:, important_variable_sets[type]]
            X_test_used = X_test.loc[:, important_variable_sets[type]]

            if model == 'svm':
                p1 = working_path + '/model_weights/' + ticker +'svmnon.h5'
            elif model == 'lr':
                p1 = working_path + '/model_weights/' + ticker + 'lr.h5'
            elif model == 'lda':
                p1 = working_path + '/model_weights/' + ticker + 'lda.h5'
            elif model == 'qda':
                p1 = working_path + '/model_weights/' + ticker + 'qda.h5'
            elif model == 'rf':
                p1 = working_path + '/model_weights/' + ticker + 'rf.h5'
            elif model == 'xgb':
                p1 = working_path + '/model_weights/' + ticker +'xgb.h5'

            if model == 'lstm':
                # Use previous best hyper parameters obtained from model trained on train and use val1 as validation set
                base_model = tf.keras.models.load_model(working_path + '/LSTM_for_FX/LSTM_' + ticker + '.h5')
                # Retrain the model with the new filtered dataset
                base_model.fit(X_train_total, y_train_total)
                # Predict on the test set
                prob = base_model.predict(X_test_used)
                test_predict = (prob > 0.5).astype(int)
            else:
                # Use previous best hyper parameters obtained from model trained on train and use val1 as validation set
                base_model = load(p1)
                # Retrain the model with the new filtered dataset
                base_model.fit(X_train_total, y_train_total)
                # Predict on the test set
                test_predict = base_model.predict(X_test_used)

            # Store evaluation metrics
            # Add accuracy scores
            accuracy_scores.append(accuracy_score(y_test, test_predict))
            # Add balanced accuracy scores
            balanced_accuracy.append(balanced_accuracy_score(y_test, test_predict))
            # Add F1 scores
            f1_scores.append(f1_score(y_test, test_predict))
            # Store the feature importance type
            type_ls.append(type)

    # Create DataFrames for different metrics and concatenate them
    num = len(important_variable_sets)
    df_accuracy = pd.DataFrame(list(zip(list(np.repeat(ticker_names,num)), accuracy_scores)),columns=['Market','Accuracy'])
    df_balanced_accuracy = pd.DataFrame(balanced_accuracy,columns=['Balanced Accuracy'])
    df_f1 = pd.DataFrame(f1_scores,columns=['F1 Score'])
    df_type = pd.DataFrame(type_ls,columns=['Type'])
    df_all = pd.concat([df_all, df_type, df_accuracy, df_balanced_accuracy, df_f1], axis=1)

    df_all.to_csv(working_path + '/Total_Results.csv', index=False)

    # Group the final DataFrame by the feature importance type
    groups = df_all.groupby('Type')
    # Create separate DataFrames for each feature importance type
    dfs = {name: data for name, data in groups}
    # Initialize these dataframes to None
    lime_df = None
    pfi_df = None
    xgb_df = None
    self_selected_df = None
    if sp_lime_top_n != None:
        lime_df = dfs['LIME']
        lime_df.to_csv(working_path + '/Results_SP_LIME.csv', index=False)
    if pfi_top_n != None:
        pfi_df = dfs['PFI']
        pfi_df.to_csv(working_path + '/Results_PFI.csv', index=False)
    if self_selected_feature_dic != None:
        self_selected_df = dfs['Self_Selected']
        self_selected_df.to_csv(working_path + '/Results_Self_Selected.csv', index=False)
    if xgbfi_gain_top_n != None:
        xgb_df = dfs['XGB']
        xgb_df.to_csv(working_path + '/Results_XGBFI.csv', index=False)

    # Generates a table for a specific type from a given dataframe and calculates ranks for each evaluation metric (Accuracy, Balanced Accuracy, F1 Score)
    tables = {}
    types = df_all['Type'].unique()
    for type_name in types:
        tables[type_name] = generate_table_for_type(df_all, type_name)

    # Save the table to a xlsx file
    path1 = working_path + '/interpret_methods_rank.xlsx'
    with pd.ExcelWriter(path1) as writer:
        for type_name, table in tables.items():
            table.to_excel(writer, sheet_name=type_name, index=False)

    return df_all, lime_df, pfi_df, xgb_df, self_selected_df, tables

