#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import subprocess
libraries_to_install = ['quantstats', 'xlsxwriter']

for library in libraries_to_install:
    subprocess.run(["pip", "install", library], check=True)


# In[ ]:


import Data_Processing as dp
import Machine_Learning_Models as ml
import quantstats as qs
import pandas as pd
import numpy as np
import seaborn as sns
import math
import multiprocessing
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import gridspec
from datetime import date
import glob, os
from joblib import dump, load
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, balanced_accuracy_score
import xgboost as xgb
import time
from collections import defaultdict
import openpyxl


# # XGB Model

# In[ ]:


def xgb_Strategy(ticker = 'USDEUR', feature_dic = None, upload = False, path = '/FX-Data/',
                         train_start = '2005-01-03', train_end = '2011-12-30',
                         train1_start = '2012-01-02', train1_end = '2014-12-31',
                         train2_start = '2015-01-01', train2_end = '2017-12-29',
                         vali_start = '2018-01-01', vali_end = '2018-12-31',
                         test_start = '2019-01-01', test_end = '2019-12-31',
                         standarlization = True, from_beginning = True, Max_depth = [3,5,7,9],
                         Min_child_weight = [1, 2, 4], Gamma = [0, 0.1, 0.2], Subsample = [0.6, 0.8, 1],
                         Colsample_bytree = [0.6, 0.8, 1], Learning_rate = [0.001, 0.01, 0.1]):

    """
    Trains and validates an XGBoost model for a given ticker for build trading strategy.

    Parameters:
    - ticker (str): Ticker symbol, default is 'USDEUR'.
    - feature_dic (dict or None): Dictionary of features for the given ticker. If None, no feature selection is done.
    - upload (bool): Whether to load a pre-trained model or not.
    - path (str): Path to data, default is '/FX-Data/'.
    - train_start (str): Start date for the training dataset.
    - train_end (str): End date for the training dataset.
    - train1_start (str): Start date for the first additional training dataset (was previously used as the first validation dataset).
    - train1_end (str): End date for the first additional training dataset (was previously used as the first validation dataset).
    - train2_start (str): Start date for the second additional training dataset (was previously used as the second validation dataset).
    - train2_end (str): End date for the second additional training dataset (was previously used as the second validation dataset).
    - vali_start (str): Start date for the validation dataset (was previously used as testing dataset).
    - vali_end (str): End date for the validation dataset (was previously used as testing dataset).
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
    - y_pred: Contains the binary predictions.
    - y_prob: Contains the probability of each prediction.
    """

    if feature_dic is not None and not isinstance(feature_dic, dict):
        raise ValueError("feature_dic must be of type dict or None.")
    # Data split
    X_train, X_train1, X_train2, X_val, y_train, y_train1, y_train2, y_val = dp.getProcessedData(ticker, path,
                                                                             train_start, train_end,
                                                                             train1_start, train1_end,
                                                                             train2_start, train2_end,
                                                                             vali_start, vali_end,
                                                                             standarlization, from_beginning)
    # Data with n days lag from test_start to test_end
    X_test_strategy, y_test_strategy = dp.getProcessedData_strategy_model(ticker, path, test_start, test_end, standarlization)

    # Combining different training datasets for better generalization
    X_train_concat = pd.concat([X_train, X_train1, X_train2])
    y_train_concat = pd.concat([y_train, y_train1, y_train2])

    # Combining training data for final model training
    X_train_final = pd.concat([X_train, X_train1, X_train2, X_val])
    y_train_final = pd.concat([y_train, y_train1, y_train2, y_val])

    working_path = os.getcwd()
    if feature_dic != None:
        # For each market, we only use several top features selected by SFFS.
        X_train_concat = X_train_concat.iloc[:,feature_dic[ticker]]
        X_train_final = X_train_final.iloc[:,feature_dic[ticker]]
        X_val = X_val.iloc[:,feature_dic[ticker]]
        X_test_strategy = X_test_strategy.iloc[:,feature_dic[ticker]]

    # Check if the model should be loaded directly
    if upload == True:
        # For ensure that the model does not change when it is used again and is easy to reproduce.
        xgb_model = load(working_path + '/Strategy_XGB_weight/' + ticker +'xgb_strategy.h5')
        y_pred = xgb_model.predict(X_test_strategy)
        y_prob = xgb_model.predict_proba(X_test_strategy)
    # Otherwise, train the model from scratch
    elif upload == False:
        best_score = 0
        # Grid search to find the best hyperparameters for the XGBoost
        # 'max_depth': Maximum depth of a tree
        for max_depth in [3,5,7,9]:
            # 'min_child_weight': Minimum sum of instance weight needed in a child
            for min_child_weight in [1, 2, 4]:
                # 'gamma': Regularization parameter
                for gamma in [0, 0.1, 0.2]:
                    # 'subsample': Proportion of training data to grow trees and prevent overfitting
                    for subsample in [0.6, 0.8, 1]:
                        # 'colsample_bytree': Subsample ratio of columns when constructing each tree
                        for colsample_bytree in [0.6, 0.8, 1]:
                            # 'learning_rate': Shrinks the feature weights to make the boosting process more conservative
                            for learning_rate in [0.001, 0.01, 0.1]:
                                xgb_model = xgb.XGBClassifier(learning_rate=learning_rate, max_depth=max_depth,
                                                              min_child_weight=min_child_weight, gamma=gamma,
                                                              subsample=subsample, colsample_bytree=colsample_bytree,
                                                              use_label_encoder=False, eval_metric='logloss',
                                                              n_estimators = int(np.sqrt(len(X_train_concat.columns))))
                                xgb_model.fit(X_train_concat, y_train_concat)
                                score = xgb_model.score(X_val, y_val)
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
                                      n_estimators = int(np.sqrt(len(X_train_concat.columns))))
        xgb_model.fit(X_train_final, y_train_final)

        # Saving the model
        path1 = working_path + '/Strategy_XGB_weight/' + ticker +'xgb_strategy.h5'
        os.makedirs(os.path.dirname(path1), exist_ok=True)
        dump(xgb_model, path1)

        y_pred = xgb_model.predict(X_test_strategy)
        y_prob = xgb_model.predict_proba(X_test_strategy)

    return y_pred, y_prob


# In[ ]:


def xgb_result(ticker_names = ['USDEUR', 'USDJPY', 'USDGBP', 'USDCHF', 'USDNZD', 'USDCAD', 'USDSEK', 'USDDKK', 'USDNOK',
                                  'EURJPY', 'EURGBP', 'EURCHF', 'EURNZD', 'EURCAD', 'EURSEK', 'EURDKK', 'EURNOK'],
                         feature_dic = None, upload = False, path = '/FX-Data/',
                         train_start = '2005-01-03', train_end = '2011-12-30',
                         train1_start = '2012-01-02', train1_end = '2014-12-31',
                         train2_start = '2015-01-01', train2_end = '2017-12-29',
                         vali_start = '2018-01-01', vali_end = '2018-12-31',
                         test_start = '2019-01-01', test_end = '2019-12-31',
                         standarlization = True, from_beginning = True, Max_depth = [3,5,7,9],
                         Min_child_weight = [1, 2, 4], Gamma = [0, 0.1, 0.2], Subsample = [0.6, 0.8, 1],
                         Colsample_bytree = [0.6, 0.8, 1], Learning_rate = [0.001, 0.01, 0.1]):

    """
    This function applies the xgb_Strategy to multiple tickers and returns the model's predictions.

    Parameters:
    - ticker_names (list): List of currency ticker names.
    - feature_dic (dict or None): Dictionary of features for the given ticker. If None, no feature selection is done.
    - upload (bool): Whether to load a pre-trained model or not.
    - path (str): Path to data, default is '/FX-Data/'.
    - train_start (str): Start date for the training dataset.
    - train_end (str): End date for the training dataset.
    - train1_start (str): Start date for the first additional training dataset (was previously used as the first validation dataset).
    - train1_end (str): End date for the first additional training dataset (was previously used as the first validation dataset).
    - train2_start (str): Start date for the second additional training dataset (was previously used as the second validation dataset).
    - train2_end (str): End date for the second additional training dataset (was previously used as the second validation dataset).
    - vali_start (str): Start date for the validation dataset (was previously used as testing dataset).
    - vali_end (str): End date for the validation dataset (was previously used as testing dataset).
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
    - fx_tic_prob (dict): Dictionary with tickers as keys and their respective binary predictions as values.
    - fx_tic_pred (dict): Dictionary with tickers as keys and their respective probability of each prediction as values.
    - Sign (dict): Dictionary with tickers as keys and their respective sign (-1 for downtick and 1 for uptick) as values.
    """

    if feature_dic is not None and not isinstance(feature_dic, dict):
        raise ValueError("feature_dic must be of type dict or None.")

    fx_tic_pred = {}
    fx_tic_prob = {}

    # Loop through each ticker
    for ticker in ticker_names:
        # Get predictions and probabilities for the current ticker using xgb_Strategy function
        test_predict, model_prob = xgb_Strategy(ticker = ticker, feature_dic = feature_dic, upload = upload, path = path,
                         train_start = train_start, train_end = train_end,
                         train1_start = train1_start, train1_end = train1_end,
                         train2_start = train2_start, train2_end = train2_end,
                         vali_start = vali_start, vali_end = vali_end,
                         test_start = test_start, test_end = test_end,
                         standarlization = standarlization, from_beginning = from_beginning, Max_depth = Max_depth,
                         Min_child_weight = Min_child_weight, Gamma = Gamma, Subsample = Subsample,
                         Colsample_bytree = Colsample_bytree, Learning_rate = Learning_rate)
        # Store the predictions and probabilities in their respective dictionaries
        fx_tic_prob[ticker] = model_prob
        fx_tic_pred[ticker] = test_predict

    # Replace all occurrences of '0' with '-1' in the predictions
    # Since we choose to short when we predict that the stock will downtick
    Sign = {ticker: np.where(arr==0, -1, arr) for ticker, arr in fx_tic_pred.items()}

    return fx_tic_prob, fx_tic_pred, Sign



# # Profit and loss of the binary strategy

# In[ ]:


def Profit_and_Loss(ticker_names = ['USDEUR', 'USDJPY', 'USDGBP', 'USDCHF', 'USDNZD', 'USDCAD', 'USDSEK', 'USDDKK', 'USDNOK',
                                  'EURJPY', 'EURGBP', 'EURCHF', 'EURNZD', 'EURCAD', 'EURSEK', 'EURDKK', 'EURNOK'],
                         feature_dic = None, upload = False, path = '/FX-Data/',
                         train_start = '2005-01-03', train_end = '2011-12-30',
                         train1_start = '2012-01-02', train1_end = '2014-12-31',
                         train2_start = '2015-01-01', train2_end = '2017-12-29',
                         vali_start = '2018-01-01', vali_end = '2018-12-31',
                         test_start = '2019-01-01', test_end = '2019-12-31',
                         standarlization = True, from_beginning = True, Max_depth = [3,5,7,9],
                         Min_child_weight = [1, 2, 4], Gamma = [0, 0.1, 0.2], Subsample = [0.6, 0.8, 1],
                         Colsample_bytree = [0.6, 0.8, 1], Learning_rate = [0.001, 0.01, 0.1],
                         leverage = 100, lag = 5):

    """
    This function computes the Profit and Loss for a set of currency tickers based on predictions generated by the xgb_Strategy and plots their performances over a specified test period.

    Parameters:
    - ticker_names (list): List of currency ticker names.
    - feature_dic (dict or None): Dictionary of features for the given ticker. If None, no feature selection is done.
    - upload (bool): Whether to load a pre-trained model or not.
    - path (str): Path to data, default is '/FX-Data/'.
    - train_start (str): Start date for the training dataset.
    - train_end (str): End date for the training dataset.
    - train1_start (str): Start date for the first additional training dataset (was previously used as the first validation dataset).
    - train1_end (str): End date for the first additional training dataset (was previously used as the first validation dataset).
    - train2_start (str): Start date for the second additional training dataset (was previously used as the second validation dataset).
    - train2_end (str): End date for the second additional training dataset (was previously used as the second validation dataset).
    - vali_start (str): Start date for the validation dataset (was previously used as testing dataset).
    - vali_end (str): End date for the validation dataset (was previously used as testing dataset).
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
    - leverage (int): Leverage for Options, default is 100.
    - lag (int): Number of lags to offset, default is 5.
    
    Returns:
    - fx_df_dic (dict): Dictionary with tickers as keys and their respective dataframe including model returns and performance metrics as values.
    """

    if feature_dic is not None and not isinstance(feature_dic, dict):
        raise ValueError("feature_dic must be of type dict or None.")

    # Get predictions, probabilities and signs for the current ticker using xgb_result function
    fx_tic_prob, fx_tic_pred, Sign = xgb_result(ticker_names = ticker_names,
                         feature_dic = feature_dic, upload = upload, path = path,
                         train_start = train_start, train_end = train_end,
                         train1_start = train1_start, train1_end = train1_end,
                         train2_start = train2_start, train2_end = train2_end,
                         vali_start = vali_start, vali_end = vali_end,
                         test_start = test_start, test_end = test_end,
                         standarlization = standarlization, from_beginning = from_beginning, Max_depth = Max_depth,
                         Min_child_weight = Min_child_weight, Gamma = Gamma, Subsample = Subsample,
                         Colsample_bytree = Colsample_bytree, Learning_rate = Learning_rate)
    working_path = os.getcwd()
    # Set the number of subplots per figure
    subplots_per_figure = 6

    fx_df_dic = {}

    for idx, ticker in enumerate(ticker_names, start=1):
        if (idx - 1) % subplots_per_figure == 0:
            remaining_tickers = len(ticker_names) - idx + 1
            num_rows = min((remaining_tickers + 1) // 2, 3)

            # Start a new figure with calculated number of rows
            fig, axs = plt.subplots(num_rows, 2, figsize=(16, 10))

            if remaining_tickers % 2 and remaining_tickers < subplots_per_figure:
                fig.delaxes(axs[-1, 1])


        i = (idx - 1) % subplots_per_figure // 2
        j = (idx - 1) % subplots_per_figure % 2
        # Calculate the subplot to plot on
        ax = axs[i, j]

        # Data split
        # Data without n days lag from test_start to test_end
        data, y1 = dp.getProcessedData_strategy_profit(ticker, path, lag, test_start, test_end)

        # Add columns for Position and Returns to the dataframe
        data['Position'] = Sign[ticker]
        # We apply leverage here
        data['Returns'] = ((data['Adj Close']/data['Open'])-1)*leverage
        data['Model_Returns'] = data['Position'] * data['Returns']

        # Store the processed data in the fx_df_dic dictionary
        fx_df_dic[ticker] = data.copy()

        # Calculate cumulative and average returns for the model
        model_cumulative_returns = ((data['Model_Returns']).cumsum())*100
        model_average_returns = (sum(data['Model_Returns'])/len(data['Model_Returns']))*100

        # Convert the index to datetime type
        model_cumulative_returns.index = pd.to_datetime(model_cumulative_returns.index)

        # Define date range with monthly frequency
        date_range = pd.date_range(start=test_start, end=test_end, freq='M')

        # Plotting the model's cumulative returns
        ax.plot(model_cumulative_returns, label = f'{ticker}', color = 'blue')
        ax.grid(True)
        ax.legend()
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Returns')
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.set_title(f'The performance of the Binary Strategy: {ticker}') # Include ticker in title

        # Formatting the x-axis to show monthly ticks
        ax.set_xticks(date_range)
        ax.set_xticklabels(date_range.strftime('%b-%Y'), rotation=45)

        ax.set_xlim(model_cumulative_returns.index.min(),model_cumulative_returns.index.max())

        # Adjusting the y-axis limits for better visualization
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        new_y_min = y_min - 0.15 * y_range
        ax.set_ylim(new_y_min, y_max)

        # Display the final Profit and Loss on the graph
        text_str = f'Profit and Loss : {model_cumulative_returns.iloc[-1]/100:.4f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.2)
        ax.text(0.35, 0.05, text_str, transform=ax.transAxes, fontsize=11, verticalalignment='bottom', bbox=props)

        # If this is the last subplot for this figure, or the last subplot overall, display the figure
        if (idx % subplots_per_figure == 0) or (idx == len(ticker_names)):

            plt.tight_layout()
            plt.savefig(f"{working_path}/PnL_figure_{(idx // subplots_per_figure) + 1}.png", bbox_inches='tight')
            plt.show()

    # Process the dataframes in fx_df_dic dictionary to ensure date is set as the index and the time zone is naive
    for ticker in ticker_names:

        fx_df_dic[ticker].reset_index(inplace = True)
        fx_df_dic[ticker]['Date'] = pd.to_datetime(fx_df_dic[ticker]['Date']).dt.tz_localize(None)
        fx_df_dic[ticker].set_index('Date', inplace = True)

    return fx_df_dic


# # Sharpe Ratio for each month

# In[ ]:


def Monthly_Sharpe_Ratio(ticker_names = ['USDEUR', 'USDJPY', 'USDGBP', 'USDCHF', 'USDNZD', 'USDCAD', 'USDSEK', 'USDDKK', 'USDNOK',
                                  'EURJPY', 'EURGBP', 'EURCHF', 'EURNZD', 'EURCAD', 'EURSEK', 'EURDKK', 'EURNOK'],
                         feature_dic = None, upload = False, path = '/FX-Data/',
                         train_start = '2005-01-03', train_end = '2011-12-30',
                         train1_start = '2012-01-02', train1_end = '2014-12-31',
                         train2_start = '2015-01-01', train2_end = '2017-12-29',
                         vali_start = '2018-01-01', vali_end = '2018-12-31',
                         test_start = '2019-01-01', test_end = '2019-12-31',
                         standarlization = True, from_beginning = True, Max_depth = [3,5,7,9],
                         Min_child_weight = [1, 2, 4], Gamma = [0, 0.1, 0.2], Subsample = [0.6, 0.8, 1],
                         Colsample_bytree = [0.6, 0.8, 1], Learning_rate = [0.001, 0.01, 0.1],
                         leverage = 100, lag = 5,
                         SOFR = [0.0247, 0.0240, 0.0242, 0.0248, 0.0242, 0.0239, 0.0245, 0.0213, 0.0227, 0.0186, 0.0157, 0.0155]):

    """
    Calculate the Monthly Sharpe Ratio for a list of tickers.

    Note: we use SOFR interest rates as risk free rate, original data published at https://www.newyorkfed.org/markets/reference-rates/sofr-averages-and-index, in daily
          monthly data get from https://www.global-rates.com/en/interest-rates/sofr/2019.aspx
    
    Parameters:
    - ticker_names (list): List of currency ticker names.
    - feature_dic (dict or None): Dictionary of features for the given ticker. If None, no feature selection is done.
    - upload (bool): Whether to load a pre-trained model or not.
    - path (str): Path to data, default is '/FX-Data/'.
    - train_start (str): Start date for the training dataset.
    - train_end (str): End date for the training dataset.
    - train1_start (str): Start date for the first additional training dataset (was previously used as the first validation dataset).
    - train1_end (str): End date for the first additional training dataset (was previously used as the first validation dataset).
    - train2_start (str): Start date for the second additional training dataset (was previously used as the second validation dataset).
    - train2_end (str): End date for the second additional training dataset (was previously used as the second validation dataset).
    - vali_start (str): Start date for the validation dataset (was previously used as testing dataset).
    - vali_end (str): End date for the validation dataset (was previously used as testing dataset).
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
    - leverage (int): Leverage for Options, default is 100.
    - lag (int): Number of lags to offset, default is 5.
    - SOFR (int): List of monthly SOFR interest rates (used as risk-free rate).
    
    Returns:
    - Monthly_sharpe_ratio (dict): Dictionary with monthly Sharpe ratios for each ticker
    - df (pd.DataFrame): Dataframe containing the monthly Sharpe ratios for each ticker
    """

    if feature_dic is not None and not isinstance(feature_dic, dict):
        raise ValueError("feature_dic must be of type dict or None.")

    # Get dictionary with tickers as keys and their respective dataframe including model returns and performance metrics as values using Profit_and_Loss function
    fx_df_dic = Profit_and_Loss(ticker_names = ticker_names,
                         feature_dic = feature_dic, upload = upload, path = path,
                         train_start = train_start, train_end = train_end,
                         train1_start = train1_start, train1_end = train1_end,
                         train2_start = train2_start, train2_end = train2_end,
                         vali_start = vali_start, vali_end = vali_end,
                         test_start = test_start, test_end = test_end,
                         standarlization = standarlization, from_beginning = from_beginning, Max_depth = Max_depth,
                         Min_child_weight = Min_child_weight, Gamma = Gamma, Subsample = Subsample,
                         Colsample_bytree = Colsample_bytree, Learning_rate = Learning_rate,
                         leverage = leverage, lag = lag)

    working_path = os.getcwd()
    # Dictionary to store the Sharpe ratio for each ticker on a monthly basis
    Monthly_sharpe_ratio = {}

    # Loop through each ticker
    for ticker in ticker_names:

        model_returns = fx_df_dic[ticker]['Model_Returns']

        # Ensure both the time series are timezone naive
        if model_returns.index.tz is not None:
            model_returns.index = model_returns.index.tz_localize(None)

        # Dictionary to store the model returns grouped by month
        monthly_dfs = {}
        # Grouping the model returns by month
        grouped = model_returns.groupby(pd.Grouper(freq='M'))

        df_name = []

        # Extract data for each month and store in dictionary
        for month, data in grouped:
            month_str = month.strftime('%Y-%m')
            df_name.append(month_str)
            monthly_dfs[month_str] = data

        monthly_sharpe_ratios = {}
        months_count = len(grouped)
        if len(SOFR) < months_count:
            raise ValueError(f"SOFR length is {len(SOFR)} but there are {months_count} months in the dataset. Ensure SOFR has enough values.")

        # Calculate Sharpe ratio for each month
        for i in range(months_count):
            # Risk-free rate for the current month
            rf = SOFR[i]
            # Month (e.g., "2019-01")
            name = df_name[i]
            strategy_returns = monthly_dfs[name]

            # Calculate monthly Sharpe ratio without annualizing
            sharpe_ratio = qs.stats.sharpe(strategy_returns, rf=rf, periods=12, annualize=False)

            monthly_sharpe_ratios[name] = sharpe_ratio

        # Store the monthly Sharpe ratios for the current ticker
        Monthly_sharpe_ratio[ticker] = monthly_sharpe_ratios
        p1 = working_path + '/Monthly_Sharpe_Ratio.h5'
        # Save the Monthly sharpe ratio to a h5 file
        dump(Monthly_sharpe_ratio, p1)
        df = pd.DataFrame(Monthly_sharpe_ratio)
        df.to_csv(working_path + '/Sharpe_Table.csv', index=True)

    return Monthly_sharpe_ratio, df


# # Monthly accumulative profit

# In[ ]:


def Monthly_accumulative_profit(ticker_names = ['USDEUR', 'USDJPY', 'USDGBP', 'USDCHF', 'USDNZD', 'USDCAD', 'USDSEK', 'USDDKK', 'USDNOK',
                                  'EURJPY', 'EURGBP', 'EURCHF', 'EURNZD', 'EURCAD', 'EURSEK', 'EURDKK', 'EURNOK'],
                         feature_dic = None, upload = False, path = '/FX-Data/',
                         train_start = '2005-01-03', train_end = '2011-12-30',
                         train1_start = '2012-01-02', train1_end = '2014-12-31',
                         train2_start = '2015-01-01', train2_end = '2017-12-29',
                         vali_start = '2018-01-01', vali_end = '2018-12-31',
                         test_start = '2019-01-01', test_end = '2019-12-31',
                         standarlization = True, from_beginning = True, Max_depth = [3,5,7,9],
                         Min_child_weight = [1, 2, 4], Gamma = [0, 0.1, 0.2], Subsample = [0.6, 0.8, 1],
                         Colsample_bytree = [0.6, 0.8, 1], Learning_rate = [0.001, 0.01, 0.1],
                         leverage = 100, lag = 5):

    """
    Calculate the Monthly Accumulative Profit for a list of tickers.
    
    Parameters:
    - ticker_names (list): List of currency ticker names.
    - feature_dic (dict or None): Dictionary of features for the given ticker. If None, no feature selection is done.
    - upload (bool): Whether to load a pre-trained model or not.
    - path (str): Path to data, default is '/FX-Data/'.
    - train_start (str): Start date for the training dataset.
    - train_end (str): End date for the training dataset.
    - train1_start (str): Start date for the first additional training dataset (was previously used as the first validation dataset).
    - train1_end (str): End date for the first additional training dataset (was previously used as the first validation dataset).
    - train2_start (str): Start date for the second additional training dataset (was previously used as the second validation dataset).
    - train2_end (str): End date for the second additional training dataset (was previously used as the second validation dataset).
    - vali_start (str): Start date for the validation dataset (was previously used as testing dataset).
    - vali_end (str): End date for the validation dataset (was previously used as testing dataset).
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
    - leverage (int): Leverage for Options, default is 100.
    - lag (int): Number of lags to offset, default is 5.
    
    Returns:
    - cum_monthly_return_dict (dict): Dictionary with monthly accumulative profits for each ticker
    - df_cum (pd.DataFrame): Dataframe containing the monthly accumulative profits for each ticker
    """

    if feature_dic is not None and not isinstance(feature_dic, dict):
        raise ValueError("feature_dic must be of type dict or None.")

    # Get dictionary with tickers as keys and their respective dataframe including model returns and performance metrics as values using Profit_and_Loss function
    fx_df_dic = Profit_and_Loss(ticker_names = ticker_names,
                         feature_dic = feature_dic, upload = upload, path = path,
                         train_start = train_start, train_end = train_end,
                         train1_start = train1_start, train1_end = train1_end,
                         train2_start = train2_start, train2_end = train2_end,
                         vali_start = vali_start, vali_end = vali_end,
                         test_start = test_start, test_end = test_end,
                         standarlization = standarlization, from_beginning = from_beginning, Max_depth = Max_depth,
                         Min_child_weight = Min_child_weight, Gamma = Gamma, Subsample = Subsample,
                         Colsample_bytree = Colsample_bytree, Learning_rate = Learning_rate,
                         leverage = leverage, lag = lag)

    working_path = os.getcwd()
    # Dictionary to store the cumulative monthly returns for each market
    cum_monthly_return_dict = {}

    # Loop through each market and resample its data to compute monthly returns
    for market, df1 in fx_df_dic.items():

        # Summing up all returns within each month
        monthly_df = df1.resample('M')['Model_Returns'].sum()
        # Convert the date index to string format
        monthly_df.index = monthly_df.index.strftime('%Y-%m')
        # Store the monthly returns in the dictionary
        cum_monthly_return_dict[market] = monthly_df.to_dict()

    df_cum = pd.DataFrame(cum_monthly_return_dict)
    df_cum.to_csv(working_path + '/Monthly_Accumulative_Profit.csv', index=True)

    return cum_monthly_return_dict, df_cum


# # Best and worst month

# In[ ]:


def BnW_Month(ticker_names = ['USDEUR', 'USDJPY', 'USDGBP', 'USDCHF', 'USDNZD', 'USDCAD', 'USDSEK', 'USDDKK', 'USDNOK',
                                  'EURJPY', 'EURGBP', 'EURCHF', 'EURNZD', 'EURCAD', 'EURSEK', 'EURDKK', 'EURNOK'],
                         feature_dic = None, upload = False, path = '/FX-Data/',
                         train_start = '2005-01-03', train_end = '2011-12-30',
                         train1_start = '2012-01-02', train1_end = '2014-12-31',
                         train2_start = '2015-01-01', train2_end = '2017-12-29',
                         vali_start = '2018-01-01', vali_end = '2018-12-31',
                         test_start = '2019-01-01', test_end = '2019-12-31',
                         standarlization = True, from_beginning = True, Max_depth = [3,5,7,9],
                         Min_child_weight = [1, 2, 4], Gamma = [0, 0.1, 0.2], Subsample = [0.6, 0.8, 1],
                         Colsample_bytree = [0.6, 0.8, 1], Learning_rate = [0.001, 0.01, 0.1],
                         leverage = 100, lag = 5,
                         SOFR = [0.0247, 0.0240, 0.0242, 0.0248, 0.0242, 0.0239, 0.0245, 0.0213, 0.0227, 0.0186, 0.0157, 0.0155],
                         visualization = True):

    """
    Compute Best and Worst monthly Permutation Feature Importance (PFI) for given tickers and visualize the results.

    Note: we use SOFR interest rates as risk free rate, original data published at https://www.newyorkfed.org/markets/reference-rates/sofr-averages-and-index, in daily
          monthly data get from https://www.global-rates.com/en/interest-rates/sofr/2019.aspx
    
    Parameters:
    - ticker_names (list): List of currency ticker names.
    - feature_dic (dict or None): Dictionary of features for the given ticker. If None, no feature selection is done.
    - upload (bool): Whether to load a pre-trained model or not.
    - path (str): Path to data, default is '/FX-Data/'.
    - train_start (str): Start date for the training dataset.
    - train_end (str): End date for the training dataset.
    - train1_start (str): Start date for the first additional training dataset (was previously used as the first validation dataset).
    - train1_end (str): End date for the first additional training dataset (was previously used as the first validation dataset).
    - train2_start (str): Start date for the second additional training dataset (was previously used as the second validation dataset).
    - train2_end (str): End date for the second additional training dataset (was previously used as the second validation dataset).
    - vali_start (str): Start date for the validation dataset (was previously used as testing dataset).
    - vali_end (str): End date for the validation dataset (was previously used as testing dataset).
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
    - leverage (int): Leverage for Options, default is 100.
    - lag (int): Number of lags to offset, default is 5.
    - SOFR (int): List of monthly SOFR interest rates (used as risk-free rate).
    - visualization (bool): If set to True, visualization plots will be displayed.

    Returns:
    - res_all_pairs (dict): Dictionary containing the sorted permutation importance values for the best and worst month of each ticker.
    """

    if feature_dic is not None and not isinstance(feature_dic, dict):
        raise ValueError("feature_dic must be of type dict or None.")

    # Get monthly sharpe ratio table using Monthly_Sharpe_Ratio function
    Monthly_sharpe_ratio, df_final = Monthly_Sharpe_Ratio(ticker_names = ticker_names,
                         feature_dic = feature_dic, upload = upload, path = path,
                         train_start = train_start, train_end = train_end,
                         train1_start = train1_start, train1_end = train1_end,
                         train2_start = train2_start, train2_end = train2_end,
                         vali_start = vali_start, vali_end = vali_end,
                         test_start = test_start, test_end = test_end,
                         standarlization = standarlization, from_beginning = from_beginning, Max_depth = Max_depth,
                         Min_child_weight = Min_child_weight, Gamma = Gamma, Subsample = Subsample,
                         Colsample_bytree = Colsample_bytree, Learning_rate = Learning_rate,
                         leverage = leverage, lag = lag,
                         SOFR = SOFR)

    working_path = os.getcwd()
    # Initialize a dictionary to store the result for best and worst months for each market
    result = {}

    # Iterate through each market and determine the best and worst month based on values in df_final
    for market in df_final.columns:
        result[market] = {
            # Fetch the index of the maximum value
            'Best': df_final[market].idxmax(),
            # Fetch the index of the minimum value
            'Worst': df_final[market].idxmin(),
        }

    # Convert the result dictionary into a DataFrame
    BnW = pd.DataFrame(result)
    BnW.to_csv(working_path + '/BnW_Month.csv', index=True)

    # Dictionary to store the best datasets
    best_datasets = {}
    # Dictionary to store the worst datasets
    worst_datasets = {}

    # Loop through each ticker and extract data for best and worst months
    for market, month_dict in result.items():
        # Data split
        # # Data with n days lag from test_start to test_end
        X_test1, y_test1 = dp.getProcessedData_strategy_model(market, path, test_start, test_end, standarlization)

        if feature_dic != None:
            # For each market, we only use several top features selected by SFFS.
            X_test_pfi = X_test1.iloc[:,feature_dic[market]]
            # Convert the index to datetime
            X_test_pfi.index = pd.to_datetime(X_test_pfi.index)
            y_test1.index = pd.to_datetime(y_test1.index)

        # Convert the best and worst month strings to datetime format
        best_month = pd.to_datetime(month_dict['Best'])
        worst_month = pd.to_datetime(month_dict['Worst'])

        # Create masks to filter the data for the best and worst months
        best_mask = (X_test_pfi.index.year == best_month.year) & (X_test_pfi.index.month == best_month.month)
        worst_mask = (X_test_pfi.index.year == worst_month.year) & (X_test_pfi.index.month == worst_month.month)

        # Filter the data based on the created masks
        best_X = X_test_pfi.loc[best_mask, ]
        best_Y = y_test1[best_mask]

        worst_X = X_test_pfi.loc[worst_mask, ]
        worst_Y = y_test1[worst_mask]

        # Store the filtered data in the respective dictionaries
        best_datasets[market] = (best_X, best_Y)
        worst_datasets[market] = (worst_X, worst_Y)

    # Dictionary to store results for each ticker
    res_all_pairs = {}

    # Determine the number of CPU cores for parallel computation
    n_cores = multiprocessing.cpu_count()

    # Loop through each ticker
    for ticker in ticker_names:
        # Initialize the dictionary to store results for current ticker
        res_all_pairs[ticker] = {'Best': {}, 'Worst': {}}

        # Extract the test datasets for the current ticker
        X_test_best = best_datasets[ticker][0]
        X_test_worst = worst_datasets[ticker][0]

        y_test_best = best_datasets[ticker][1]
        y_test_worst = worst_datasets[ticker][1]

        # Load the previously trained model with best hyperparameters for the current ticker
        xgb_model = load(working_path + '/Strategy_XGB_weight/' + ticker +'xgb_strategy.h5')

        # Calculate permutation importance for the best dataset
        result_best = permutation_importance(xgb_model, X_test_best, y_test_best, n_jobs = n_cores, n_repeats=5, random_state=42)
        # Calculate permutation importance for the worst dataset
        result_worst = permutation_importance(xgb_model, X_test_worst, y_test_worst, n_jobs = n_cores, n_repeats=5, random_state=42)

        # Store the average PFI score for each feature in a dictionary
        result_global_pfi_best = {feature: pfi_score for feature, pfi_score in zip(X_test_best.columns, result_best.importances_mean)}
        result_global_pfi_worst = {feature: pfi_score for feature, pfi_score in zip(X_test_worst.columns, result_worst.importances_mean)}

        # Sort and store the results in the final dictionary
        res_all_pairs[ticker]['Best'] = sorted(result_global_pfi_best.items(), key=lambda x: x[1], reverse=True)
        res_all_pairs[ticker]['Worst'] = sorted(result_global_pfi_worst.items(), key=lambda x: x[1], reverse=True)


    # Save the results to a h5 file
    dump(res_all_pairs, working_path + '/Strategy_PFI_BnW_Month.h5')

    # Output path for saving the results in Excel format
    output_path = working_path + '/Strategy_PFI_BnW_Month.xlsx'
    # Prepare the writer to save data in Excel format
    writer = pd.ExcelWriter(output_path, engine='xlsxwriter')
    # Convert and save the data in Excel sheets
    for currency, datasets in res_all_pairs.items():
        for dataset, pfi_values in datasets.items():
            # Consider only features with positive PFI values
            positive_pfi_values = [val for val in pfi_values if val[1] > 0]

            # If no feature has positive PFI value, store a placeholder message
            if not positive_pfi_values:
                df = pd.DataFrame([0], columns=["None features are important"], index=[dataset])
            else:
                # Convert the results into a DataFrame for saving
                df = pd.DataFrame(positive_pfi_values, columns=["Feature", "PFI"]).set_index("Feature").T

            # Save the data in a new sheet
            df.to_excel(writer, sheet_name=f"{currency}_{dataset}")

    # Save the Excel file
    writer.save()

    if visualization == True:
        # Model visualization
        for ticker1, pfi_data in res_all_pairs.items():
            for dataset1, pfi_scores in pfi_data.items():
                # Extract the feature names and corresponding scores
                features, scores = zip(*pfi_scores)
                # Order the features by their scores
                sorted_idx = np.argsort(scores)

                # Plotting
                fig, ax = plt.subplots()
                # Scores are the importances and features are the corresponding labels
                ax.barh(np.array(features)[sorted_idx], np.array(scores)[sorted_idx])
                ax.set_title(f"Permutation Importances for {ticker1} in {dataset1} Month")
                fig.tight_layout()
                plt.show()

    return res_all_pairs


# # Best and worst accumulative month

# In[ ]:


def max_subarray(data):

    """
    Find the contiguous subarray (one or more numbers) which has the maximum sum.
    
    Parameters:
    - data (list or array): The input data containing numeric values.
    
    Returns:
    - max_sum (float): Maximum sum of the subarray.
    - start (int): Starting index of the subarray.
    - end (int): Ending index of the subarray.
    """

    # Initialize max_sum to negative infinity
    max_sum = -np.inf
    # Current sum
    temp_sum = 0
    start = 0
    end = 0
    temp_start = 0

    # Iterate over the array
    for i in range(len(data)):
        if temp_sum <= 0:
            temp_sum = data[i]
            temp_start = i
        else:
            temp_sum += data[i]
        # Update maximum sum if current sum becomes larger
        if temp_sum > max_sum:
            max_sum = temp_sum
            start = temp_start
            end = i

    return max_sum, start, end


# In[ ]:


def min_subarray(data):

    """
    Find the contiguous subarray (one or more numbers) which has the minimum sum.
    
    Parameters:
    - data (list or array): The input data containing numeric values.
    
    Returns:
    - min_sum (float): Minimum sum of the subarray.
    - start (int): Starting index of the subarray.
    - end (int): Ending index of the subarray.
    """
    # Initialize min_sum to positive infinity
    min_sum = np.inf
    # Current sum
    temp_sum = 0
    start = 0
    end = 0
    temp_start = 0

    # Iterate over the array
    for i in range(len(data)):
        if temp_sum >= 0:
            temp_sum = data[i]
            temp_start = i
        else:
            temp_sum += data[i]
        # Update minimum sum if current sum becomes smaller
        if temp_sum < min_sum:
            min_sum = temp_sum
            start = temp_start
            end = i

    return min_sum, start, end


# In[ ]:


def BnW_Cum_Month(ticker_names = ['USDEUR', 'USDJPY', 'USDGBP', 'USDCHF', 'USDNZD', 'USDCAD', 'USDSEK', 'USDDKK', 'USDNOK',
                                  'EURJPY', 'EURGBP', 'EURCHF', 'EURNZD', 'EURCAD', 'EURSEK', 'EURDKK', 'EURNOK'],
                         feature_dic = None, upload = False, path = '/FX-Data/',
                         train_start = '2005-01-03', train_end = '2011-12-30',
                         train1_start = '2012-01-02', train1_end = '2014-12-31',
                         train2_start = '2015-01-01', train2_end = '2017-12-29',
                         vali_start = '2018-01-01', vali_end = '2018-12-31',
                         test_start = '2019-01-01', test_end = '2019-12-31',
                         standarlization = True, from_beginning = True, Max_depth = [3,5,7,9],
                         Min_child_weight = [1, 2, 4], Gamma = [0, 0.1, 0.2], Subsample = [0.6, 0.8, 1],
                         Colsample_bytree = [0.6, 0.8, 1], Learning_rate = [0.001, 0.01, 0.1],
                         leverage = 100, lag = 5,
                         visualization = True):
                
    """
    Function to evaluate the best and worst cumulative month for given tickers.
    
    Parameters:
    - ticker_names (list): List of currency ticker names.
    - feature_dic (dict or None): Dictionary of features for the given ticker. If None, no feature selection is done.
    - upload (bool): Whether to load a pre-trained model or not.
    - path (str): Path to data, default is '/FX-Data/'.
    - train_start (str): Start date for the training dataset.
    - train_end (str): End date for the training dataset.
    - train1_start (str): Start date for the first additional training dataset (was previously used as the first validation dataset).
    - train1_end (str): End date for the first additional training dataset (was previously used as the first validation dataset).
    - train2_start (str): Start date for the second additional training dataset (was previously used as the second validation dataset).
    - train2_end (str): End date for the second additional training dataset (was previously used as the second validation dataset).
    - vali_start (str): Start date for the validation dataset (was previously used as testing dataset).
    - vali_end (str): End date for the validation dataset (was previously used as testing dataset).
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
    - leverage (int): Leverage for Options, default is 100.
    - lag (int): Number of lags to offset, default is 5.
    - visualization (bool): If set to True, visualization plots will be displayed.
    
    Returns:
    - res_all_pairs (dict): Dictionary containing the permutation feature importances for each ticker for the best and worst months.
    """

    if feature_dic is not None and not isinstance(feature_dic, dict):
        raise ValueError("feature_dic must be of type dict or None.")

    # Get cumulative monthly return table using Monthly_accumulative_profit function
    cum_monthly_return_dict, df_final = Monthly_accumulative_profit(ticker_names = ticker_names,
                         feature_dic = feature_dic, upload = upload, path = path,
                         train_start = train_start, train_end = train_end,
                         train1_start = train1_start, train1_end = train1_end,
                         train2_start = train2_start, train2_end = train2_end,
                         vali_start = vali_start, vali_end = vali_end,
                         test_start = test_start, test_end = test_end,
                         standarlization = standarlization, from_beginning = from_beginning, Max_depth = Max_depth,
                         Min_child_weight = Min_child_weight, Gamma = Gamma, Subsample = Subsample,
                         Colsample_bytree = Colsample_bytree, Learning_rate = Learning_rate,
                         leverage = leverage, lag = lag)

    working_path = os.getcwd()
    # Compute max and min subarray for each column in df_final
    results = {}

    for col in df_final.columns:
        data = df_final[col].values
        max_profit, max_start, max_end = max_subarray(data)
        min_profit, min_start, min_end = min_subarray(data)

        results[col] = {
            'max_profit': max_profit,
            'max_months': (df_final.index[max_start], df_final.index[max_end]),
            'min_profit': min_profit,
            'min_months': (df_final.index[min_start], df_final.index[min_end]),
        }

    # Display the results for each market
    for market0, res0 in results.items():
        print(f"{market0}:")
        print(f"  Max profit: {res0['max_profit']}, from {res0['max_months'][0]} to {res0['max_months'][1]}")
        print(f"  Min profit: {res0['min_profit']}, from {res0['min_months'][0]} to {res0['min_months'][1]}")

    # Dictionary to store the max datasets
    max_datasets = {}
    # Dictionary to store the min datasets
    min_datasets = {}
    # Loop through each ticker and extract data for max and min cumulative months
    for market, res in results.items():
        # Data split
        # Data with n days lag from test_start to test_end
        X_test1, y_test1 = dp.getProcessedData_strategy_model(market, path, test_start, test_end, standarlization)

        if feature_dic != None:
            # For each market, we only use several top features selected by SFFS.
            X_test_pfi = X_test1.iloc[:,feature_dic[market]]
            X_test_pfi.index = pd.to_datetime(X_test_pfi.index)
            y_test1.index = pd.to_datetime(y_test1.index)

        # Convert the max and min cumulative months strings to datetime format
        max_start, max_end = pd.to_datetime(res['max_months'])
        min_start, min_end = pd.to_datetime(res['min_months'])

        # Adjust end month to cover the entire month
        max_end = max_end + pd.offsets.MonthEnd(1)
        min_end = min_end + pd.offsets.MonthEnd(1)

        # Create masks to filter the data for the max and min cumulative months
        max_mask = (X_test_pfi.index >= max_start) & (X_test_pfi.index <= max_end)
        min_mask = (X_test_pfi.index >= min_start) & (X_test_pfi.index <= min_end)

        # Filter the data based on the created masks
        max_X = X_test_pfi.loc[max_mask, ]
        max_Y = y_test1[max_mask]

        min_X = X_test_pfi.loc[min_mask, ]
        min_Y = y_test1[min_mask]

        # Store the filtered data in the respective dictionaries
        max_datasets[market] = (max_X, max_Y)
        min_datasets[market] = (min_X, min_Y)

    # Dictionary to store results for each ticker
    res_all_pairs = {}

    # Determine the number of CPU cores for parallel computation
    n_cores = multiprocessing.cpu_count()

    # Loop through each ticker
    for ticker in ticker_names:
        # Initialize the dictionary to store results for current ticker
        res_all_pairs[ticker] = {'Max': {}, 'Min': {}}

        # Extract the test datasets for the current ticker
        X_test_best = max_datasets[ticker][0]
        X_test_worst = min_datasets[ticker][0]

        y_test_best = max_datasets[ticker][1]
        y_test_worst = min_datasets[ticker][1]

        # Load the previously trained model with best hyperparameters for the current ticker
        xgb_model = load(working_path + '/Strategy_XGB_weight/' + ticker +'xgb_strategy.h5')

        # Calculate permutation importance for the max dataset
        result_best = permutation_importance(xgb_model, X_test_best, y_test_best, n_jobs = n_cores, n_repeats=5, random_state=42)
        # Calculate permutation importance for the min dataset
        result_worst = permutation_importance(xgb_model, X_test_worst, y_test_worst, n_jobs = n_cores, n_repeats=5, random_state=42)

        # Store the average PFI score for each feature in a dictionary
        result_global_pfi_best = {feature: pfi_score for feature, pfi_score in zip(X_test_best.columns, result_best.importances_mean)}
        result_global_pfi_worst = {feature: pfi_score for feature, pfi_score in zip(X_test_worst.columns, result_worst.importances_mean)}

        # Sort and store the results in the final dictionary
        res_all_pairs[ticker]['Max'] = sorted(result_global_pfi_best.items(), key=lambda x: x[1], reverse=True)
        res_all_pairs[ticker]['Min'] = sorted(result_global_pfi_worst.items(), key=lambda x: x[1], reverse=True)

    # Save the results to a h5 file
    dump(res_all_pairs, working_path + '/Strategy_PFI_Cum_Month.h5')

    # Output path for saving the results in Excel format
    output_path = working_path + '/Strategy_PFI_Cum_Month.xlsx'

    # Prepare the writer to save data in Excel format
    writer = pd.ExcelWriter(output_path, engine='xlsxwriter')
    # Convert and save the data in Excel sheets
    for currency, datasets in res_all_pairs.items():
        for dataset, pfi_values in datasets.items():
            # Consider only features with positive PFI values
            positive_pfi_values = [val for val in pfi_values if val[1] > 0]

            # If no feature has positive PFI value, store a placeholder message
            if not positive_pfi_values:
                df = pd.DataFrame([0], columns=["None features are important"], index=[dataset])
            else:
                # Convert the results into a DataFrame for saving
                df = pd.DataFrame(positive_pfi_values, columns=["Feature", "PFI"]).set_index("Feature").T

            # Save the data in a new sheet
            df.to_excel(writer, sheet_name=f"{currency}_{dataset}")

    # Save the Excel file
    writer.save()

    if visualization == True:
        # model visualization
        for ticker1, pfi_data in res_all_pairs.items():
            for dataset1, pfi_scores in pfi_data.items():
                # Extract the feature names and corresponding scores
                features, scores = zip(*pfi_scores)
                # Order the features by their scores
                sorted_idx = np.argsort(scores)

                # Plotting
                fig, ax = plt.subplots()
                # Scores are the importances and features are the corresponding labels
                ax.barh(np.array(features)[sorted_idx], np.array(scores)[sorted_idx])
                ax.set_title(f"Permutation Importances for {ticker1} in {dataset1} Accumulative Months")
                fig.tight_layout()
                plt.show()

    return res_all_pairs

