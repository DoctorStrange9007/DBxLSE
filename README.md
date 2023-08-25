# P3_FeatureImportance_2022_23

A detailed repository that contains the data, code, model weights and output results used in **The Art of Interpretation: Unleashing the Potential for
Insightful Predictions in FX Data**, also includes the pipeline we built to help researchers conduct broader research in the field. 

In addition, we provide a method to reproduce our research results in the pipeline section.

The following are the file names and locations in this repository of the codes and files corresponding to each part of our research.

## Table of Contents
- [Report](#report)
- [Data](#data)
- [Codes](#codes)
  - [Part 0: Data preprocessing](#part0-data-preprocessing)
  - [Part 1: Interpretability methods](#part1-interpretability-methods)
  - [Part 2: Trading strategy](#part2-trading-strategy)
- [Model weights](#model-weights)
- [Output](#output)
- [Pipeline (Module)](#pipeline-module)


---

## **Report**

        https://github.com/DoctorStrange9007/DBxLSE/blob/main/GROUP3capstone.pdf

---

## **Data**

This section contains all datasets used in the research.

FX-Data

        https://github.com/DoctorStrange9007/DBxLSE/tree/main/FX-Data

---
## **Codes**

### **Part0: Data preprocessing**

- **Section 1**: Data download and feature engineering

    F1_data_download.ipynb

        https://github.com/DoctorStrange9007/DBxLSE/blob/main/F1_data_download.ipynb

- **Section 2(1)**: Exploratory data analysis

    F2a_EDA.ipynb

        https://github.com/DoctorStrange9007/DBxLSE/blob/main/F2a_EDA.ipynb

- **Section 2(2)**: Feature selection with SFFS (Sequential Forward Feature Selection)

    F2b_Feature_Selection.ipynb

        https://github.com/DoctorStrange9007/DBxLSE/blob/main/F2b_Feature_Selection.ipynb

- **General**: Contains all datasets split and processing functions that needed in follow-up research

    Data_Preparation.ipynb

        https://github.com/DoctorStrange9007/DBxLSE/blob/main/Data_Preparation.ipynb

---

### **Part1: Interpretability methods**

- **Section 1(1)**: Common non-deep learning models for classification

    P1a1_Statistical_Learning_Models.ipynb

        https://github.com/DoctorStrange9007/DBxLSE/blob/main/P1a1_Statistical_Learning_Models.ipynb

- **Section 1(2)**: LSTM Model

    P1a2_LSTM_Model.ipynb

        https://github.com/DoctorStrange9007/DBxLSE/blob/main/P1a2_LSTM_Model.ipynb

- **Section 2(1)**: PFI (Permutation Feature Importance)

    P1b1_PFI.ipynb

        https://github.com/DoctorStrange9007/DBxLSE/blob/main/P1b1_PFI.ipynb

- **Section 2(2)**: SP LIME (Submodular Pick Local Interpretable Model-agnostic Explanations)

    P1b2_SP_LIME.ipynb

        https://github.com/DoctorStrange9007/DBxLSE/blob/main/P1b2_SP_LIME.ipynb

- **Section 2(3)**: XGBoost Built-in Feature Importance, calculated by the average gain across all splits the feature is used in

    P1b3_XGB_Feature_Importance.ipynb

        https://github.com/DoctorStrange9007/DBxLSE/blob/main/P1b3_XGB_Feature_Importance.ipynb

- **Section 3(1)**: Comparing the above interpretability methods in a specific perspective

    P1c1_SP_LIMEvsPFIvsRandomvsXGB.ipynb

        https://github.com/DoctorStrange9007/DBxLSE/blob/main/P1c1_SP_LIMEvsPFIvsRandomvsXGB.ipynb

- **Section 3(2)**: Concatenate and visualize the machine learning method performance evaluation form, and visualize the research results of the interpretability method section

    P1c2_table.ipynb

        https://github.com/DoctorStrange9007/DBxLSE/blob/main/P1c2_table.ipynb

---

### **Part2: Trading strategy**

- **Section 1**: Trading model construction, plotting model performance, calculating monthly Sharpe ratio, and monthly cumulative return

    P2a_Strategy_PnL.ipynb

        https://github.com/DoctorStrange9007/DBxLSE/blob/main/P2a_Strategy_PnL.ipynb

- **Section 2(1)**: Find the best and worst months in each market based on the Sharpe ratio and use PFI to analyze which technical indicators play major roles in the best and worst months

    P2b1_BnW_Month_PFI.ipynb

        https://github.com/DoctorStrange9007/DBxLSE/blob/main/P2b1_BnW_Month_PFI.ipynb

- **Section 2(2)**: Find the best and worst performing periods in each market based on the monthly cumulative return and use PFI to analyze which technical indicatorsplay major roles in the best performing period and the worst performing period

    P2b2_BnW_Cumulative_Month_PFI.ipynb

        https://github.com/DoctorStrange9007/DBxLSE/blob/main/P2b2_BnW_Cumulative_Month_PFI.ipynb

---

## **Model weights**

Model weights, stored in the h5 files

- **Non-deep learning machine lerning models weights**

    model_weights

        https://github.com/DoctorStrange9007/DBxLSE/tree/main/model_weights

- **LSTM weights**

    LSTM_for_FX

        https://github.com/DoctorStrange9007/DBxLSE/tree/main/LSTM_for_FX

- **XGBoost weights (for trading strategy part)**

    Strategy_XGB_weight

        https://github.com/DoctorStrange9007/DBxLSE/tree/main/Strategy_XGB_weight

---

## **Output**

- **EDA results**

    Excels_and_CSVs/statistics_df.csv

        https://github.com/DoctorStrange9007/DBxLSE/blob/main/Excels_and_CSVs/statistics_df.csv

    Excels_and_CSVs/statistics_df_advance.csv

        https://github.com/DoctorStrange9007/DBxLSE/blob/main/Excels_and_CSVs/statistics_df_advance.csv

- **Non-deep learning machine lerning models evaluation**

    Excels_and_CSVs/ml_acc_df_new_data.csv

        https://github.com/DoctorStrange9007/DBxLSE/blob/main/Excels_and_CSVs/ml_acc_df_new_data.csv

- **LSTM evaluation**

    Excels_and_CSVs/LSTM_acc_df_val1_for_vali.csv

        https://github.com/DoctorStrange9007/DBxLSE/blob/main/Excels_and_CSVs/LSTM_acc_df_val1_for_vali.csv

- **PFI results (h5 files)**

    PFI_result

        https://github.com/DoctorStrange9007/DBxLSE/tree/main/PFI_result

- **SP LIME results (h5 files)**

    SP_LIME_result_new

        https://github.com/DoctorStrange9007/DBxLSE/tree/main/SP_LIME_result_new

- **XGBoost Built-in Feature Importance results (h5 files)**

    XGB_FI_result

        https://github.com/DoctorStrange9007/DBxLSE/tree/main/XGB_FI_result

- **interpretability methods evaluation**

    Excels_and_CSVs/Part_1_result_new

        https://github.com/DoctorStrange9007/DBxLSE/tree/main/Excels_and_CSVs/Part_1_result_new

- **Monthly Sharpe ratio results**

    Excels_and_CSVs/Sharpe_table_new.csv

        https://github.com/DoctorStrange9007/DBxLSE/blob/main/Excels_and_CSVs/Sharpe_table_new.csv

- **Monthly accumulative profit results**

    Excels_and_CSVs/Monthly_accumulative_profit.csv

        https://github.com/DoctorStrange9007/DBxLSE/blob/main/Excels_and_CSVs/Monthly_accumulative_profit.csv

- **Best and worst performing month results**

    Excels_and_CSVs/BnW_Month_new.csv

        https://github.com/DoctorStrange9007/DBxLSE/blob/main/Excels_and_CSVs/BnW_Month_new.csv
  
- **PFI results for best and worst performing months** 

    Excels_and_CSVs/Strategy_PFI_BnW_Month.xlsx

        https://github.com/DoctorStrange9007/DBxLSE/blob/main/Excels_and_CSVs/Strategy_PFI_BnW_Month.xlsx

- **PFI results for best and worst performing periods** 

    Excels_and_CSVs/Strategy_PFI_Cum_Month.xlsx

        https://github.com/DoctorStrange9007/DBxLSE/blob/main/Excels_and_CSVs/Strategy_PFI_Cum_Month.xlsx
  
---

## **Pipeline (Module)**

This section contains the Pipeline (Module) and its usage.

        https://github.com/DoctorStrange9007/DBxLSE/tree/main/Pipeline
