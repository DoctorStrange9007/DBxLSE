# User Guide

## 1, Environmental requirements

To use this pipeline, you need to have the TA-Lib already installed.  


### For Mac OS X user, please follow the steps on: 

https://github.com/TA-Lib/ta-lib-python#mac-os-x:~:text=forge%20ta%2Dlib-,Dependencies,-To%20use%20TA

### For Apple Silicon user, such as the M1 processors, please follow the steps on: 

https://stackoverflow.com/questions/66056725/is-it-possible-that-mac-m1-users-are-not-able-to-use-python-wrapper-for-ta-lib#:~:text=3-,I,-was%20able%20to

### For Windows user, please follow the steps on:

https://github.com/TA-Lib/ta-lib-python#mac-os-x:~:text=dir%20ta%2Dlib-,Windows,-Download%20ta%2Dlib

### For Google Colab user, please run the follow code: 


!wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz

!tar -xzvf ta-lib-0.4.0-src.tar.gz

%cd ta-lib

!./configure --prefix=/usr

!make

!make install

%cd ..

!pip install TA-Lib

## 2, How to reproduce the results included in our report

The Example_For_Reproduce.ipynb file shows how to easily reproduce our work.

https://github.com/lse-st498/P3_FeatureImportance_2022_23/blob/main/Pipeline/Example_For_Reproduce.ipynb

The zip file contains all the relevant h5 files that will help you reproduce our results exactly. The only thing you need to do is to save all py files and zip files to your working path, then unzip those zip files.

Please note that due to the difference in the system environment, python version, python environment and library version, if you choose not to use the h5 file to import the pre-trained model but to train the relevant model by yourself, even if you use the same random seed, it may not be completely consistent results, although the differences are generally not large. This kind of problem is unavoidable, and it often appears in papers using python as the programming language，especially in deep learning related fields. But with the h5 file, you'll be using the exact same model that we're using. So please make sure to always set upload to True.

The same is true for feature importance, so when reproducing please make sure to use the features we selected from our model in our environment, this will help to ensure get the same results at each step.

Also, we did not include random in the reproduce part. This is because we used truly random features instead of pseudo-random, which makes it more natural but also lead to impossible to reproduce the exact same results. Furthermore, our core purpose is to compare three interpretability methods, and the results of all three interpretability methods are fully reproducible, so not including random in the reproduction process won't make much difference.

## 3, Wider usage

Compared with our original code, this pipeline has been greatly improved in flexibility, so it can be widely used for related further more research problems.

We will further expand its functionality and improve its flexibility in the future to ensure that it becomes a mature and easy-to-use library in this field.
