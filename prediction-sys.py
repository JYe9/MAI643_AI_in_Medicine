import pandas as pd 
import numpy as np
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from scipy.stats import loguniform
from sklearn.metrics import roc_curve, auc
from sklearn import metrics

# Visualization
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

np.set_printoptions(formatter={'float':"{:6.5g}".format})

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


risk_factor_df = pd.read_csv("risk_factors_cervical_cancer.csv", 
            na_values=["?"])

print("----------------------------------- Information -----------------------------------")
risk_factor_df.info()

print("----------------------------------- Missing Values -----------------------------------")
missing_info = risk_factor_df.isnull().sum()
total_nan = missing_info.sum()
total_entries = risk_factor_df.size

# Print total NaN values
if (total_nan == 0):
    print("\nNo NaN values in the dataset.")
else:
    print("\nNaN values found in the dataset.")

    print("\nTotal NaN values in dataset: {}/{}".format(total_nan, total_entries))

    # Sort columns by the number of missing values
    nan_columns = missing_info.sort_values(ascending=False)

    print("\nTop 15 columns with missing values:\n")
    for i, (col, count) in enumerate(nan_columns.head(15).items(), 1):
        print("{:2}. {:35} : {:}".format(i, col, count))

risk_factor_df.drop(columns=["STDs: Time since first diagnosis", "STDs: Time since last diagnosis"], inplace=True)

# Rows containing NaN values
total_rows = len(risk_factor_df)
nan_rows = risk_factor_df.isna().any(axis=1).tolist().count(True)
print("\nTotal Rows containing NaN values in dataset: {}/{}".format(nan_rows, total_rows))

# Find rows that contain more than 10 NaN values
rows_to_del = risk_factor_df[risk_factor_df.isna().sum(axis=1) > 10].index

print("\nRows containing >10 NaN values: {}/{}".format(len(rows_to_del), total_rows))

# Remove rows
risk_factor_df.drop(rows_to_del, inplace=True)
risk_factor_df.reset_index(drop=True, inplace=True)

print("--------------------------- Handling Missing Values ---------------------------")
print("----------------------------------- BEFORE -----------------------------------")
print("Number of rows before filling missing values: ", len(risk_factor_df))

# Display the number of missing values before filling
print("\nNumber of missing values per column before filling:")
print(risk_factor_df.isnull().sum())

# Fill missing values depending on the column
for col in risk_factor_df.columns:
    # If the column has more than 3 unique values, fill with mean of the column
    if risk_factor_df[col].nunique() > 3:
        risk_factor_df[col] = risk_factor_df[col].fillna(risk_factor_df[col].median())
    
# Drop rest NaN containing rows
risk_factor_df=risk_factor_df.dropna()
risk_factor_df.reset_index(drop=True, inplace=True)

print("\n----------------------------------- AFTER -----------------------------------")
print("Number of rows after filling missing values: ", len(risk_factor_df))

# Display the number of missing values after filling
print("\nNumber of missing values per column after filling:")
print(risk_factor_df.isnull().sum())

print("----------------------------------- Duplicate Rows -----------------------------------")
# Check for duplicate rows
duplicate_rows = risk_factor_df.duplicated()

# Count the number of duplicate rows
num_duplicates = duplicate_rows.sum()

if num_duplicates == 0:
    print("No duplicate rows found in the dataset.")
else:
    print(f"Found {num_duplicates} duplicate rows in the dataset.\n")

    # Display the duplicate rows indexes (if any)
    print("Duplicate rows indexes: {}\n".format(risk_factor_df[duplicate_rows].index.values))

    # Removing duplicate rows
    print("----------------------------- Removing Duplicates ----------------------------")
    print("----------------------------------- BEFORE -----------------------------------")
    print("Number of rows before removing duplicates: ", len(risk_factor_df))

    risk_factor_df.drop_duplicates(inplace=True)
    risk_factor_df.reset_index(drop=True, inplace=True)

    print("\n----------------------------------- AFTER -----------------------------------")
    print("Number of rows after removing duplicates: ", len(risk_factor_df))

mean_df = risk_factor_df.mean()
std_df = risk_factor_df.std()

# Print columns that have a standard deviation 0 (contain only one value)
print("Columns containing 1 value: {}\n".format(std_df[std_df==0].index.values))

risk_factor_df.drop(columns=["STDs:cervical condylomatosis", "STDs:AIDS"], inplace=True)

# Function finding the unique values of each column in the dataframe
def find_unique_values_df(feat: pd.DataFrame):
    return {col: feat[col].unique() for col in feat}

def find_outliers(col, indices):
    obs = risk_factor_df[col].iloc[indices]
    unique_items, counts = np.unique(obs, return_counts=True)
    unique_items, counts = unique_items[::-1], counts[::-1]

    values_to_delete = unique_items[counts < 2 ]
    return values_to_delete

def delete_outliers(col, to_delete):
    if (to_delete.size != 0):
        rows_to_del = risk_factor_df.loc[risk_factor_df[col].isin(to_delete)].index.values.tolist()

        # Remove rows
        risk_factor_df.drop(rows_to_del, inplace=True)
        risk_factor_df.reset_index(drop=True, inplace=True)
    
# Unique Values
unique_vals = find_unique_values_df(risk_factor_df)
# Identify non-binary columns
non_binary_cols = [col for col, vals in unique_vals.items() if len(vals) > 2]

for col in non_binary_cols:

    # IQR cannot be applied to columns with median 0
    if (risk_factor_df[col].median() != 0):
        Q3, Q1 = np.percentile(risk_factor_df[col], [75 ,25])
        IQR = Q3-Q1

        upper = Q3+(1.5*IQR)
        lower = Q1-(1.5*IQR)

        print(col)
        print("median: {}, upper fence: {}, lower fence: {}\n".format(risk_factor_df[col].median(), upper, lower))

        #Delete one occurrence observations outside the upper fence as outliers
        upper_to_delete = find_outliers(col, np.where(risk_factor_df[col] > upper)[0])
        delete_outliers(col, upper_to_delete)

        
        #Delete one occurrence observations outside the lower fence as outliers
        lower_to_delete = find_outliers(col, np.where(risk_factor_df[col] < lower)[0])
        delete_outliers(col, lower_to_delete)

print("\nFinal dataset size: {} cols, {} rows".format(risk_factor_df.shape[1], risk_factor_df.shape[0]))


X = risk_factor_df.drop(columns=["Hinselmann", "Schiller", "Citology", "Biopsy"])
y = risk_factor_df["Biopsy"]

X_train, X_temp, y_train, y_temp = train_test_split(X, 
                                                    y, 
                                                    test_size=0.33, 
                                                    random_state=42,
                                                    stratify = y, 
                                                    shuffle=True)
X_val, X_test, y_val, y_test = train_test_split(X_temp, 
                                                y_temp, 
                                                test_size=0.5,
                                                random_state=42,
                                                stratify = y_temp,
                                                shuffle=True)


from imblearn.under_sampling import NeighbourhoodCleaningRule
from imblearn.over_sampling import BorderlineSMOTE #0.62

nc = NeighbourhoodCleaningRule()
x_train_sm, y_train_sm = nc.fit_resample(X_train, y_train)

smote_enn = BorderlineSMOTE(random_state=0)
x_train_sm, y_train_sm = smote_enn.fit_resample(x_train_sm, y_train_sm)

scaler = StandardScaler()
# Specify the features to be scaled
features_to_scale = ["Smokes (years)", "Smokes (packs/year)", "Hormonal Contraceptives (years)", "IUD (years)"]

# Scale the specified features and add new columns with suffix '_scaled'
x_train_sm[[f"{feature}_scaled" for feature in features_to_scale]] = scaler.fit_transform(x_train_sm[features_to_scale])

# Transform the scaled columns in the validation and test sets as well
X_val[[f"{feature}_scaled" for feature in features_to_scale]] = scaler.transform(X_val[features_to_scale])
X_test[[f"{feature}_scaled" for feature in features_to_scale]] = scaler.transform(X_test[features_to_scale])

#Drop original columns from the training set
x_train_sm.drop(columns=features_to_scale, inplace=True)
X_val.drop(columns=features_to_scale, inplace=True)
X_test.drop(columns=features_to_scale, inplace=True) # 0.65

# Define age categories, every 5 years as an intervals
age_intervals = [13, 18, 23, 28, 33, 38, 43, 48, 53, 58, 63, 68, 73, 78, 83, 88]
labels = list(range(len(age_intervals) - 1))

# Encode the "Age" feature
x_train_sm['Age_encoded'] = pd.cut(x_train_sm['Age'], bins=age_intervals, labels=labels, right=False)
X_val['Age_encoded'] = pd.cut(X_val['Age'], bins=age_intervals, labels=labels, right=False)
X_test['Age_encoded'] = pd.cut(X_test['Age'], bins=age_intervals, labels=labels, right=False)

#Drop original column from the training set
x_train_sm.drop(columns=['Age'], inplace=True)
X_val.drop(columns=['Age'], inplace=True)
X_test.drop(columns=['Age'], inplace=True) # 0.64

# Scale data
minMaxScaler = MinMaxScaler()
x_train_rescaled = minMaxScaler.fit_transform(x_train_sm)
x_val_rescaled = minMaxScaler.transform(X_val)
x_test_rescaled = minMaxScaler.transform(X_test)

# 95% of variance
pca = PCA(n_components=11)
x_train_reduced = pca.fit_transform(x_train_rescaled)
x_val_reduced = pca.transform(x_val_rescaled)
x_test_reduced = pca.transform(x_test_rescaled) #0.69

clf = RandomForestClassifier(random_state=0)
clf.fit(x_train_reduced, y_train_sm)
pred_proba = clf.predict_proba(x_val_reduced)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_val, pred_proba)
auc = metrics.roc_auc_score(y_val, pred_proba)
print(auc)