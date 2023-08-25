#!/usr/bin/env python
# coding: utf-8

# In[1]:


import subprocess
libraries_to_install = ['pandas_datareader', 'yfinance', 'finta', 'pandas-ta', 'ta', 'mlxtend']

for library in libraries_to_install:
    subprocess.run(["pip", "install", library], check=True)


# In[2]:


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

install("joblib", "1.2.0")


# In[3]:


from pandas_datareader import data as pdr
from datetime import date
from datetime import datetime
import yfinance as yf
yf.pdr_override() 
import pandas as pd
import numpy as np
import ta
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
import talib
import pandas_ta
from finta import TA
import glob, os
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from joblib import dump, load, Parallel, delayed
import scipy.stats as stats
from statsmodels.stats.stattools import durbin_watson
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.evaluate import PredefinedHoldoutSplit



# # Data Download

# In[21]:


def addFeatures(dataframe, lag = 5):

    """
    Add technical indicators as features to a given dataframe.

    Parameters:
    - dataframe (pd.DataFrame): Input data with financial metrics.
    - lag (int, optional): The number of time steps to shift for prediction. Defaults to 5.

    Returns:
    - pd.DataFrame: Dataframe with added technical indicators as features.
    """

    # Technical indicators calculation using various libraries like TA, ta, talib, and pandas_ta
    dataframe['Parabolic_SAR']=ta.trend.PSARIndicator(dataframe['High'],dataframe['Low'],dataframe['Close']).psar().shift(lag)
    dataframe['Coppock_Curve']=TA.COPP(dataframe).shift(lag)
    dataframe['Typical_Price']=TA.TP(dataframe).shift(lag)

    indicator1='RSI'
    indicator2='SO'
    indicator4='WI'
    indicator5='ROC'
    indicator6='EMA'
    indicator7='CCI'
    indicator8='BB_HB'
    indicator9='BB_LB'
    indicator10='BB_MAVG'
    indicator12='DPO'
    indicator13='ULCERINDEX'
    indicator14='SMA'
    indicator15='WMA'
    indicator16='MOM'
    indicator17='DX'
    indicator18='TRIMA'
    indicator19='AROON_DOWN'
    indicator20='AROON_UP'
    indicator21='AROONOSC'
    indicator22='ADX'
    indicator23='CMO'
    indicator24='DEMA'
    indicator25='MIDPOINT'
    indicator26='MIDPRICE'
    indicator27='NATR'
    indicator28='TEMA'
    indicator29='PSL'
    indicator30='BIAS'
    indicator31='RVI'
    indicator32='LINREG'
    indicator33='ACC_LOW'
    indicator34='ACC_MID'
    indicator35='ACC_UP'
    indicator36='CHOP'
    indicator37='RVGI'
    indicator38='PERCENT_B'
    indicator39='ATR'
    indicator40='KC_UPPER'
    indicator41='KC_LOWER'
    indicator43='BBWIDTH'
    indicator44='CHANDELIER_SHORT'
    indicator45='CHANDELIER_LONG'
    indicator48='HMA'
    indicator49='KAMA'
    indicator50='MI'
    indicator51='MSD'
    indicator52='TRIX'
    indicator53='VORTEX_NEG'
    indicator54='VORTEX_POS'
    indicator100='MACD'
    indicator101='PPO'
    indicator104='APO'
    indicator106='DO_UP'


    dataframe[indicator1]=ta.momentum.rsi(dataframe['Close']).shift(lag)
    dataframe[indicator2]=ta.momentum.stoch(dataframe['High'],dataframe['Low'],dataframe['Close']).shift(lag)
    dataframe[indicator14]=ta.trend.sma_indicator(dataframe['Close']).shift(lag)
    dataframe[indicator4]=ta.momentum.williams_r(dataframe['High'],dataframe['Low'],dataframe['Close']).shift(lag)
    dataframe[indicator5]=ta.momentum.roc(dataframe['Close']).shift(lag)
    dataframe[indicator6]=ta.trend.ema_indicator(dataframe['Close']).shift(lag)
    dataframe[indicator7]=ta.trend.cci(dataframe['High'],dataframe['Low'],dataframe['Close']).shift(lag)
    dataframe[indicator8]=ta.volatility.BollingerBands(close=dataframe['Close']).bollinger_hband().shift(lag)
    dataframe[indicator9]=ta.volatility.BollingerBands(close=dataframe['Close']).bollinger_lband().shift(lag)
    dataframe[indicator10]=ta.volatility.BollingerBands(close=dataframe['Close']).bollinger_mavg().shift(lag)
    dataframe[indicator12]=ta.trend.dpo(close=dataframe['Close']).shift(lag)
    dataframe[indicator13]=ta.volatility.ulcer_index(close=dataframe['Close']).shift(lag)
    dataframe[indicator15]=talib.WMA(dataframe['Close']).shift(lag)
    dataframe[indicator16]=talib.MOM(dataframe['Close']).shift(lag)
    dataframe[indicator17]=talib.DX(dataframe['High'],dataframe['Low'],dataframe['Close']).shift(lag)
    dataframe[indicator18]=talib.TRIMA(dataframe['Close']).shift(lag)
    dataframe[indicator19]=talib.AROON(dataframe['High'], dataframe['Low'])[0].shift(lag)
    dataframe[indicator20]=talib.AROON(dataframe['High'], dataframe['Low'])[1].shift(lag)
    dataframe[indicator21]=talib.AROONOSC(dataframe['High'], dataframe['Low']).shift(lag)
    dataframe[indicator22]=talib.ADX(dataframe['High'], dataframe['Low'], dataframe['Close']).shift(lag)
    dataframe[indicator23]=talib.CMO(dataframe['Close']).shift(lag)
    dataframe[indicator24]=talib.DEMA(dataframe['Close']).shift(lag)
    dataframe[indicator25]=talib.MIDPOINT(dataframe['Close']).shift(lag)
    dataframe[indicator26]=talib.MIDPRICE(dataframe['High'], dataframe['Low']).shift(lag)
    dataframe[indicator27]=talib.NATR(dataframe['High'], dataframe['Low'], dataframe['Close']).shift(lag)
    dataframe[indicator28]=talib.TEMA(dataframe['Close']).shift(lag)
    dataframe[indicator29]=pandas_ta.psl(dataframe['Close'], dataframe['Open']).shift(lag)
    dataframe[indicator30]=pandas_ta.bias(dataframe['Close']).shift(lag)
    dataframe[indicator31]=pandas_ta.rvi(dataframe['Close'], dataframe['High'], dataframe['Low']).shift(lag)
    dataframe[indicator32]=pandas_ta.linreg(dataframe['Close']).shift(lag)
    dataframe[indicator33]=pandas_ta.accbands(dataframe['High'], dataframe['Low'], dataframe['Close']).iloc[:,0].shift(lag)
    dataframe[indicator34]=pandas_ta.accbands(dataframe['High'], dataframe['Low'], dataframe['Close']).iloc[:,1].shift(lag)
    dataframe[indicator35]=pandas_ta.accbands(dataframe['High'], dataframe['Low'], dataframe['Close']).iloc[:,2].shift(lag)
    dataframe[indicator36]=pandas_ta.chop(dataframe['High'], dataframe['Low'], dataframe['Close']).shift(lag)
    dataframe[indicator37]=pandas_ta.rvgi(dataframe['Open'], dataframe['High'], dataframe['Low'], dataframe['Close']).iloc[:,0].shift(lag)
    dataframe[indicator38]=TA.PERCENT_B(dataframe).shift(lag)
    dataframe[indicator39]=TA.ATR(dataframe).shift(lag)
    dataframe[indicator40]=TA.KC(dataframe).iloc[:,0].shift(lag)
    dataframe[indicator41]=TA.KC(dataframe).iloc[:,1].shift(lag)
    dataframe[indicator43]=TA.BBWIDTH(dataframe).shift(lag)
    dataframe[indicator44]=TA.CHANDELIER(dataframe).iloc[:,0].shift(lag)
    dataframe[indicator45]=TA.CHANDELIER(dataframe).iloc[:,1].shift(lag)
    dataframe[indicator48]=TA.HMA(dataframe).shift(lag)
    dataframe[indicator49]=TA.KAMA(dataframe).shift(lag)
    dataframe[indicator50]=TA.MI(dataframe).shift(lag)
    dataframe[indicator51]=TA.MSD(dataframe).shift(lag)
    dataframe[indicator52]=TA.TRIX(dataframe).shift(lag)
    dataframe[indicator53]=TA.VORTEX(dataframe).iloc[:,0].shift(lag)
    dataframe[indicator54]=TA.VORTEX(dataframe).iloc[:,1].shift(lag)
    dataframe[indicator100]=ta.trend.macd(dataframe['Close']).shift(lag)
    dataframe[indicator101]=ta.momentum.ppo(dataframe['Close']).shift(lag)
    dataframe[indicator104]=talib.APO(dataframe['Close']).shift(lag)
    dataframe[indicator106]=TA.DO(dataframe).iloc[:,1].shift(lag)
    
    # Shifting the core columns by the lag amount
    dataframe['Open'] = dataframe['Open'].shift(lag)
    dataframe['Adj Close'] = dataframe['Adj Close'].shift(lag)
    dataframe['Volume'] = dataframe['Volume'].shift(lag)
    dataframe['High'] = dataframe['High'].shift(lag)
    dataframe['Low'] = dataframe['Low'].shift(lag)

    # Dropping unnecessary columns  
    dataframe = dataframe.drop('Close', axis=1)
    dataframe = dataframe.drop('Volume', axis=1)

    # Filtering data from a specific date
    dataframe = dataframe.loc['2005-01-03':,:]
    # Rearranging column order
    colnames = list(dataframe.columns)
    colnames.remove('Trend')
    dataframe = dataframe.loc[:,['Trend'] + colnames]


    return dataframe

def addClassifier(dataframe):

    """
    Add a binary classifier column to a dataframe based on the price trend.

    Parameters:
    - dataframe (pd.DataFrame): Input data with financial metrics.

    Returns:
    - pd.DataFrame: Dataframe with an added classifier column.
    """

    # Create a list to store the trend data (1 for upward and 0 for downward)
    ls_temp = []

    # Calculate the trend based on the difference between opening and adjusted closing prices
    for ele in dataframe['Open']-dataframe['Adj Close']:
        if ele<=0:
            ls_temp.append(1)
        else:
            ls_temp.append(0)

    # Add the trend data to the dataframe
    dataframe['Trend'] = ls_temp

    return dataframe

def addExtraLags(dataframe, lags = [5,10]):

    """
    Add extra lagged features to a dataframe.

    Parameters:
    - dataframe (pd.DataFrame): Input data with financial metrics.
    - lags (list, optional): List of lag values to add to the dataframe. Defaults to [5, 10].

    Returns:
    - pd.DataFrame: Dataframe with added lagged features.
    """

    # Make a copy of the dataframe's columns
    dataframe_colnames = dataframe.copy()

    # For each lag value in the provided list
    for lag in lags:
        for col in dataframe_colnames.columns:
            if col == 'Trend':
                continue
            name = col+'_'+str(lag)
            # Add lagged column to the dataframe
            dataframe[name] = dataframe[col].shift(lag-1) #minus one as the column is already of lag 1

    # Filter rows based on the maximum lag
    max_lag = max(lags)
    dataframe = dataframe.iloc[max_lag-1:,:]
    return dataframe


def SaveData(df, filename, path = '/FX-Data/'):

    """
    Save a dataframe to a CSV file.

    Parameters:
    - df (pd.DataFrame): Data to be saved.
    - filename (str): Name of the file to save the data.
    - path (str, optional): Directory path to save the file. Defaults to '/FX-Data/'.

    Returns:
    None
    """
    # Get the current working directory and save the dataframe to the specified path and filename
    working_path = os.getcwd()
    df.to_csv(working_path + path + filename+'.csv')

def getData(ticker, name, lag = 5, lags = [5,10], path = '/FX-Data/', extra_lags = False):

    """
    Fetch financial data, add technical indicators and classifiers, and save the processed data to a CSV.

    Parameters:
    - ticker (str): Financial instrument identifier.
    - name (str): Name associated with the ticker.
    - lag (int, optional): The number of time steps to shift for prediction. Defaults to 5.
    - lags (list, optional): List of extra lag values to add to the dataframe. Defaults to [5, 10].
    - path (str, optional): Directory path to save the file. Defaults to '/FX-Data/'.
    - extra_lags (bool, optional): Whether to add extra lag values from the `lags` parameter. Defaults to False.

    Returns:
    None
    """

    today = date.today()
    start_date = datetime(2003, 6, 17)
    files = []
    
    # Fetch financial data using Yahoo's API
    data = pdr.get_data_yahoo(ticker, start=start_date, end=today)

    # Add classifiers and features to the fetched data
    data = addClassifier(data)
    data = addFeatures(data, lag)

    # If extra_lags is set to True, then add the extra lags
    if extra_lags == True:
        data = addExtraLags(data, lags)

    dataname= name
    files.append(dataname)
    # Save the data to a CSV file
    SaveData(data, dataname, path)

def Data_Download(ticker_names = ['USDEUR', 'USDJPY', 'USDGBP', 'USDCHF', 'USDNZD', 'USDCAD', 'USDSEK', 'USDDKK', 'USDNOK', 
                                  'EURJPY', 'EURGBP', 'EURCHF', 'EURNZD', 'EURCAD', 'EURSEK', 'EURDKK', 'EURNOK'], 
                  ticker_list = ['EUR=X', 'JPY=X', 'GBP=X', 'CHF=X', 'NZD=X', 'CAD=X', 'SEK=X', 'DKK=X', 'NOK=X', 
                                 'EURJPY=X', 'EURGBP=X', 'EURCHF=X', 'EURNZD=X', 'EURCAD=X', 'EURSEK=X', 'EURDKK=X', 'EURNOK=X'], 
                  Lag = 5, Lags = [5,10], Path = '/FX-Data/', Extra_lags = False):

    """
    Download, process, and save financial data for a list of tickers.

    Parameters:
    - ticker_names (list): List of names associated with the tickers.
    - ticker_list (list): List of financial instrument identifiers.
    - Lag (int, optional): The number of time steps to shift for prediction. Defaults to 5.
    - Lags (list, optional): List of extra lag values to add to the dataframe. Defaults to [5, 10].
    - Path (str, optional): Directory path to save the files. Defaults to '/FX-Data/'.
    - Extra_lags (bool, optional): Whether to add extra lag values from the `Lags` parameter. Defaults to False.

    Returns:
    None
    """

    # For each ticker in the list, fetch, process and save its data
    for name, ticker in zip(ticker_names, ticker_list):
        getData(ticker, name, lag = Lag, lags = Lags, path = Path, extra_lags = Extra_lags)


# # Dataset Split

# In[23]:


def preprocessing(dataframe, train_start = '2005-01-03', train_end = '2011-12-30', 
                         vali1_start = '2012-01-02', vali1_end = '2014-12-31', 
                         vali2_start = '2015-01-01', vali2_end = '2017-12-29', 
                         test_start = '2018-01-01', test_end = '2018-12-31', 
                         standarlization = True, from_beginning = True):

    """
    Split the provided dataframe into training, validation, and test sets, and optionally standardize the data within a specific date range and optionally standardize the data.
    
    Parameters:
    - dataframe (pd.DataFrame): The dataset as a pandas DataFrame.
    - train_start (str): Start date for the training dataset.
    - train_end (str): End date for the training dataset.
    - vali1_start (str): Start date for the first validation dataset.
    - vali1_end (str): End date for the first validation dataset.
    - vali2_start (str): Start date for the second validation dataset.
    - vali2_end (str): End date for the second validation dataset.
    - test_start (str): Start date for the testing dataset.
    - test_end (str): End date for the testing dataset.
    - standarlization (bool, optional): Whether or not to standardize the data. Defaults to True.
    - from_beginning (bool, optional): Whether to include all data from the beginning to train_end in the training set. Defaults to True.
    
    Returns:
    - tuple: Features and Targets for training, first validation, second validation, and test datasets within the specified date range.
    """

    # Separate the dataframe into features and targets
    X = dataframe.iloc[:,1:].copy()
    y = dataframe.iloc[:,0].copy()

    # Split the data based on the provided date ranges
    if from_beginning == True:
        X_train = X.loc[:train_end].copy()
    else:
        X_train = X.loc[train_start:train_end].copy()

    # Splitting validation and test data
    X_val1 = X.loc[vali1_start:vali1_end].copy()
    X_val2 = X.loc[vali2_start:vali2_end].copy()
    X_test = X.loc[test_start:test_end].copy()

    if from_beginning == True:
        y_train = y.loc[:train_end].copy()
    else:
        y_train = y.loc[train_start:train_end].copy()
    y_val1 = y.loc[vali1_start:vali1_end].copy()
    y_val2 = y.loc[vali2_start:vali2_end].copy()
    y_test = y.loc[test_start:test_end].copy()

    # If standardization is enabled, then standardize the data
    if standarlization == True:
        # Compute means and standard deviations for standardization
        train_mean = X_train.mean()
        train_std = X_train.std()
        val1_mean = X_val1.mean()
        val1_std = X_val1.std()
        val2_mean = X_val2.mean()
        val2_std = X_val2.std()
        test_mean = X_test.mean()
        test_std = X_test.std()

        # Apply standardization
        X_train = (X_train - train_mean)/train_std
        X_val1 = (X_val1 - val1_mean)/val1_std
        X_val2 = (X_val2 - val2_mean)/val2_std
        X_test = (X_test - test_mean)/test_std

    return X_train, X_val1, X_val2, X_test, y_train, y_val1, y_val2, y_test


def preprocessing_strategy_model(dataframe, start_date = '2019-01-01', end_date = '2019-12-31', standarlization = True):

    """
    Process data for strategy model training within a specific date range and optionally standardize the data.

    Parameters:
    - dataframe (pd.DataFrame): The dataset as a pandas DataFrame.
    - start_date (str): Start date for the dataset.
    - end_date (str): End date for the dataset.
    - standarlization (bool, optional): Whether or not to standardize the data. Defaults to True.

    Returns:
    - tuple: Features and Targets for strategy model training within the specified date range.
    """

    # Extract strategy model data within the specified date range
    X = dataframe.iloc[:,1:].copy()
    y = dataframe.iloc[:,0].copy()

    X_strategy_model = X.loc[start_date:end_date].copy()

    y_strategy_model = y.loc[start_date:end_date].copy()

    # If standardization is enabled, standardize the data
    if standarlization == True:
        # Compute means and standard deviations for standardization
        test_mean = X_strategy_model.mean()
        test_std = X_strategy_model.std()

        # Apply standardization
        X_strategy_model = (X_strategy_model - test_mean) / test_std

    return X_strategy_model, y_strategy_model


def preprocessing_strategy_profit(dataframe, lag = 5, start_date = '2019-01-01', end_date = '2019-12-31'):

    """
    Process data for strategy profit calculation by offset lags and extracting data within a specific date range.

    Parameters:
    - dataframe (pd.DataFrame): The dataset as a pandas DataFrame.
    - lag (int): Number of lags was previously added to the model.
    - start_date (str): Start date for the dataset.
    - end_date (str): End date for the dataset.

    Returns:
    - tuple: Features and Targets for strategy profit calculation within the specified date range.
    """

    X = dataframe.iloc[:,1:].copy()
    y = dataframe.iloc[:,0].copy()

    # Offset the features by the given lag
    X1 = X.shift(periods = -lag).copy()

    # Extract strategy profit data within the specified date range
    X_strategy_profit = X1.loc[start_date:end_date].copy()

    y_strategy_profit = y.loc[start_date:end_date].copy()


    return X_strategy_profit, y_strategy_profit

def getProcessedData(ticker = 'USDEUR', path = '/FX-Data/',
                         train_start = '2005-01-03', train_end = '2011-12-30', 
                         vali1_start = '2012-01-02', vali1_end = '2014-12-31', 
                         vali2_start = '2015-01-01', vali2_end = '2017-12-29', 
                         test_start = '2018-01-01', test_end = '2018-12-31', 
                         standarlization = True, from_beginning = True):

    """
    Load and preprocess data for a specific ticker within defined training, validation, and test periods within a specific date range and optionally standardize the data.

    Parameters:
    - ticker (str): The ticker for which data needs to be processed.
    - path (str): Path to the directory containing data.
    - train_start (str): Start date for the training dataset.
    - train_end (str): End date for the training dataset.
    - vali1_start (str): Start date for the first validation dataset.
    - vali1_end (str): End date for the first validation dataset.
    - vali2_start (str): Start date for the second validation dataset.
    - vali2_end (str): End date for the second validation dataset.
    - test_start (str): Start date for the testing dataset.
    - test_end (str): End date for the testing dataset.
    - standarlization (bool, optional): Whether or not to standardize the data. Defaults to True.
    - from_beginning (bool, optional): Whether to include all data from the beginning to train_end in the training set. Defaults to True.

    Returns:
    - tuple: Features and Targets for training, first validation, second validation, and test datasets within the specified date range.
    """

    # Get the current working directory
    working_path = os.getcwd()
    # Construct the full path for the data file    
    filename = working_path + path + ticker + '.csv'
    # Load the data into a pandas DataFrame
    df = pd.read_csv(filename, index_col=0)
    # Call the preprocessing function and return its results
    return preprocessing(df, train_start, train_end, 
                  vali1_start, vali1_end, 
                  vali2_start, vali2_end, 
                  test_start, test_end, 
                  standarlization, from_beginning)

def getProcessedData_strategy_model(ticker = 'USDEUR', path = '/FX-Data/',
                         start_date = '2019-01-01', end_date = '2019-12-31', standarlization = True):

    """
    Load and preprocess data for strategy model training within a specific date range and optionally standardize the data.

    Parameters:
    - ticker (str): The ticker for which data needs to be processed for strategy model.
    - path (str): Path to the directory containing data.
    - start_date (str): Start date for the dataset.
    - end_date (str): End date for the dataset.
    - standarlization (bool, optional): Whether or not to standardize the data. Defaults to True.

    Returns:
    - tuple: Features and Targets for strategy model training within the specified date range.
    """
    # Get the current working directory
    working_path = os.getcwd()  
    # Construct the full path for the data file  
    filename = working_path + path + ticker + '.csv'
    # Load the data into a pandas DataFrame
    df = pd.read_csv(filename, index_col=0)
    # Call the preprocessing_strategy_model function and return its results
    return preprocessing_strategy_model(df, start_date, end_date, standarlization)

def getProcessedData_strategy_profit(ticker = 'USDEUR', path = '/FX-Data/',
                         lag = 5, start_date = '2019-01-01', end_date = '2019-12-31'):

    """
    Load and preprocess data for strategy profit calculation by offset lags and extracting data within a specific date range.

    Parameters:
    - ticker (str): The ticker for which data needs to be processed for strategy profit.
    - path (str): Path to the directory containing data.
    - lag (int): Number of lags to be added to the dataframe.
    - start_date (str): Start date for the dataset.
    - end_date (str): End date for the dataset.

    Returns:
    - tuple: Features and Targets for strategy profit calculation within the specified date range.
    """

    # Get the current working directory
    working_path = os.getcwd()    
    # Construct the full path for the data file
    filename = working_path + path + ticker + '.csv'
    # Load the data into a pandas DataFrame
    df = pd.read_csv(filename, index_col=0)
    # Call the preprocessing_strategy_profit function and return its results
    return preprocessing_strategy_profit(df, lag, start_date, end_date)


# # Feature Selection

# In[31]:


def sffs_with_lda(ticker = 'USDEUR', features = 20, cross_validation = 0, path = '/FX-Data/',
                         train_start = '2005-01-03', train_end = '2011-12-30', 
                         vali1_start = '2012-01-02', vali1_end = '2014-12-31', 
                         vali2_start = '2015-01-01', vali2_end = '2017-12-29', 
                         test_start = '2018-01-01', test_end = '2018-12-31', 
                         standarlization = True, from_beginning = True):


    """
    Perform Sequential Forward Feature Selection (SFFS) with Linear Discriminant Analysis (LDA) as estimator.

    Parameters:
    - ticker (str): Currency ticker name.
    - features (int): Number of features to select.
    - cross_validation (int): Number of cross-validation folds.
    - path (str): Directory path for the FX data.
    - train_start (str): Start date for the training dataset.
    - train_end (str): End date for the training dataset.
    - vali1_start (str): Start date for the first validation dataset.
    - vali1_end (str): End date for the first validation dataset.
    - vali2_start (str): Start date for the second validation dataset.
    - vali2_end (str): End date for the second validation dataset.
    - test_start (str): Start date for the testing dataset.
    - test_end (str): End date for the testing dataset.
    - standarlization (bool): Whether to standardize the data.
    - from_beginning (bool): Whether to start the process from the beginning.

    Returns:
    - list: Indices of the selected features.
    - float: Score of the selected feature set.
    - object: Fitted SFFS object.
    """

    # Get the processed data using the provided parameters
    X_train, X_val1, X_val2, X_test, y_train, y_val1, y_val2, y_test = getProcessedData(ticker, path,
                         train_start, train_end, vali1_start, vali1_end, vali2_start, vali2_end, 
                         test_start, test_end, standarlization, from_beginning)

    # Instantiate the LDA estimator
    lda = LinearDiscriminantAnalysis()
    lda1 = lda.fit(X_train, y_train)
    # Initialize the SFFS with the desired parameters
    sffs = SFS(lda1,
           k_features=features,
           forward=True,
           floating=True,
           scoring='accuracy',
           cv=cross_validation,
           n_jobs=-1)

    # Fit SFFS with the training data
    sffs = sffs.fit(X_train, y_train)

    return list(sffs.k_feature_idx_), sffs.k_score_, sffs


def sffs_with_logist(ticker = 'USDEUR', features = 20, cross_validation = 0, path = '/FX-Data/',
                         train_start = '2005-01-03', train_end = '2011-12-30', 
                         vali1_start = '2012-01-02', vali1_end = '2014-12-31', 
                         vali2_start = '2015-01-01', vali2_end = '2017-12-29', 
                         test_start = '2018-01-01', test_end = '2018-12-31', 
                         standarlization = True, from_beginning = True):

    """
    Perform Sequential Forward Feature Selection (SFFS) with Logistic Regression as estimator.

    Parameters:
    - ticker (str): Currency ticker name.
    - features (int): Number of features to select.
    - cross_validation (int): Number of cross-validation folds.
    - path (str): Directory path for the FX data.
    - train_start (str): Start date for the training dataset.
    - train_end (str): End date for the training dataset.
    - vali1_start (str): Start date for the first validation dataset.
    - vali1_end (str): End date for the first validation dataset.
    - vali2_start (str): Start date for the second validation dataset.
    - vali2_end (str): End date for the second validation dataset.
    - test_start (str): Start date for the testing dataset.
    - test_end (str): End date for the testing dataset.
    - standarlization (bool): Whether to standardize the data.
    - from_beginning (bool): Whether to start the process from the beginning.

    Returns:
    - list: Indices of the selected features.
    - float: Score of the selected feature set.
    - object: Fitted SFFS object.
    """

    # Get the processed data using the provided parameters
    X_train, X_val1, X_val2, X_test, y_train, y_val1, y_val2, y_test = getProcessedData(ticker, path,
                         train_start, train_end, vali1_start, vali1_end, vali2_start, vali2_end, 
                         test_start, test_end, standarlization, from_beginning)
    
    # Instantiate the Logistic Regression estimator
    lg = LogisticRegression()
    lg1 = lg.fit(X_train, y_train)
    # Initialize the SFFS with the desired parameters
    sffs = SFS(lg1,
           k_features=features,
           forward=True,
           floating=True,
           scoring='accuracy',
           cv=cross_validation,
           n_jobs=-1)

    # Fit SFFS with the training data
    sffs = sffs.fit(X_train, y_train)

    return list(sffs.k_feature_idx_), sffs.k_score_, sffs


def Feature_Selection(ticker_names = ['USDEUR', 'USDJPY', 'USDGBP', 'USDCHF', 'USDNZD', 'USDCAD', 'USDSEK', 'USDDKK', 'USDNOK', 
                                  'EURJPY', 'EURGBP', 'EURCHF', 'EURNZD', 'EURCAD', 'EURSEK', 'EURDKK', 'EURNOK'], 
                  features = 20, cross_validation = 0, LDA = True, path = '/FX-Data/',
                         train_start = '2005-01-03', train_end = '2011-12-30', 
                         vali1_start = '2012-01-02', vali1_end = '2014-12-31', 
                         vali2_start = '2015-01-01', vali2_end = '2017-12-29', 
                         test_start = '2018-01-01', test_end = '2018-12-31', 
                         standarlization = True, from_beginning = True):

    """
    Perform feature selection for a list of tickers using either LDA or Logistic Regression.

    Parameters:

    - ticker_names (list): List of currency ticker names.
    - features (int): Number of features to select.
    - cross_validation (int): Number of cross-validation folds.
    - LDA (bool): Use LDA if True, otherwise use Logistic Regression.
    - path (str): Directory path for the FX data.
    - train_start (str): Start date for the training dataset.
    - train_end (str): End date for the training dataset.
    - vali1_start (str): Start date for the first validation dataset.
    - vali1_end (str): End date for the first validation dataset.
    - vali2_start (str): Start date for the second validation dataset.
    - vali2_end (str): End date for the second validation dataset.
    - test_start (str): Start date for the testing dataset.
    - test_end (str): End date for the testing dataset.
    - standarlization (bool): Whether to standardize the data.
    - from_beginning (bool): Whether to start the process from the beginning.

    Returns:
    - dict: Dictionary with ticker names as keys and list of selected feature indices as values.
    """

    # Dictionary to store selected features for each ticker
    feature_dic = {}
    if LDA == True:
        # If LDA is chosen, apply SFFS with LDA for each ticker
        for ticker in ticker_names:
            feature_subset = sffs_with_lda(ticker, features, cross_validation, path,
                         train_start, train_end, vali1_start, vali1_end, vali2_start, vali2_end, 
                         test_start, test_end, standarlization, from_beginning)
            feature_dic[ticker] = feature_subset[0] 
            print(ticker)
            print(feature_subset[0])
            print(feature_subset[1])
    else:
        # If Logistic Regression is chosen, apply SFFS with Logistic Regression for each ticker
        for ticker in ticker_names:
            feature_subset = sffs_with_logist(ticker, features, cross_validation, path,
                         train_start, train_end, vali1_start, vali1_end, vali2_start, vali2_end, 
                         test_start, test_end, standarlization, from_beginning)
            feature_dic[ticker] = feature_subset[0]
            print(ticker)
            print(feature_subset[0])
            print(feature_subset[1])
    return feature_dic


# # EDA

# In[ ]:

def EDA(path = '/FX-Data', start_date = None, end_date = '2019-12-31', advance = False):

    """
    Perform Exploratory Data Analysis (EDA) on the FX data.

    Parameters:
    - path (str): Directory path for the FX data.
    - start_date (str): Starting date for EDA.
    - end_date (str): Ending date for EDA.
    - advance (bool): Compute advanced statistical metrics if True.

    Returns:
    - DataFrame: Dataframe containing computed statistical metrics for the data.
    """

    # Set the directory path for FX data
    working_path = os.getcwd()    
    folder_path = working_path + path

    # List all CSV files in the directory
    all_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.csv')]

    # Dictionary to hold each file's dataframe
    dfs = {}

    for file in all_files:
        # The key will be the filename without .csv and the value will be the dataframe
        dfs[file.split('.csv')[0]] = pd.read_csv(os.path.join(folder_path, file))

    # Initialize a DataFrame to store statistics about each file's data
    if advance == True:
        # If advance flag is set, compute detailed statistical metrics
        statistics_df = pd.DataFrame(columns=['Name', 'Mean', 'Std', 'Min', 'Q1', 'Median', 'Q3', 'Max', 'Mode', 'Skewness', 'Kurtosis', 'K-S test p-value', 'Durbin-Watson statistic'])
    else:
        # Otherwise, compute basic metrics
        statistics_df = pd.DataFrame(columns=['Name', 'Mean', 'Std', 'Skewness', 'Kurtosis', 'K-S test p-value', 'Durbin-Watson statistic'])

    # Iterate over each dataframe to compute the desired metrics
    for df_name, df in dfs.items():
        # Filter the DataFrame to only consider data we used and get the 'Adj Close' column
        if start_date != None:
            subset = df[start_date <= df['Date'] <= end_date]['Adj Close']
        else:
            subset = df[df['Date'] <= end_date]['Adj Close']

        # Compute basic statistical metrics
        mean = subset.mean()
        std = subset.std()
        skewness = subset.skew()
        kurtosis = subset.kurt()
        # If advance flag is set, compute advanced metrics as well
        if advance == True:
            min_val = subset.min()
            max_val = subset.max()
            q1 = subset.quantile(0.25)
            median = subset.quantile(0.5)
            q3 = subset.quantile(0.75)
            mode = subset.mode().iloc[0]

        # Standardize the data for the Kolmogorov–Smirnov test (zero mean and unit variance)
        standardized_data = (subset - mean) / std

        # Perform the Kolmogorov–Smirnov test against a normal distribution
        ks_stat, ks_p_value = stats.kstest(standardized_data, 'norm')

        # Calculate the Durbin–Watson statistic to detect the presence of autocorrelation
        dw_stat = durbin_watson(subset)

        if advance == True:
        # Append the results to the statistics DataFrame
            statistics_df = statistics_df.append({
                'Name': df_name,
                'Mean': mean,
                'Std': std,
                'Min': min_val,
                'Q1': q1,
                'Median': median,
                'Q3': q3,
                'Max': max_val,
                'Mode': mode,
                'Skewness': skewness,
                'Kurtosis': kurtosis,
                'K-S test p-value': ks_p_value,
                'Durbin-Watson statistic': dw_stat
            }, ignore_index=True)
        else:
            statistics_df = statistics_df.append({
                'Name': df_name,
                'Mean': mean,
                'Std': std,
                'Skewness': skewness,
                'Kurtosis': kurtosis,
                'K-S test p-value': ks_p_value,
                'Durbin-Watson statistic': dw_stat
            }, ignore_index=True)

    return statistics_df



