# Project Assignment for UCY MAI643 Artificial Intelligence in Medicine

This repository contains the source code for the group assignment for the **MAI643 - AI in Medicine** module. For our assignment, we used the **"Cervical Cancer (Risk Factors)"** dataset to develop a cervical cancer prediction system. 

We used Python for the development of the system and employed various machine learning libraries such as scikit-learn, imblearn, xgboost and plotly for the visualization of the results.

## Documentation

We split the development of the system into three main phases. The analysis phase, the preprocessing phase and the development phase.

Durinf the analysis phases we explored the dataset and employed various visualization techniques to gain a better understanding of it. The code used for this phase is contained in the `preliminary-analysis.ipynb` file.  

Next, we preprocessed the dataset. During this phase, we handled the missing values within the dataset, as well as removed duplicate rows. Additionally, we excluded certain features based on their variance and addressed the outliers in each feature. Following that, we split the dataset into 3 sets: a training set, a validation set and a test set, containing 70%, 15%, and 15% of the observations, respectively. Lastly, we performed some further data cleaning to the training set to aid the machine learning models to generalize better and achieve better results. Namely, we addressed the imbalance of the target variable "Biopsy", we scaled and encoded several features and we performed dimensionality reduction using PCA. The code used for this phase of the development is contained in the `pre-processing.ipynb` file.

## Documentation

We split the development of the system into three main phases. The analysis phase, the preprocessing phase and the development phase.

### Preliminary Analysis
Durinf the analysis phases we explored the dataset and employed various visualization techniques to gain a better understanding of it. The code used for this phase is contained in the `preliminary-analysis.ipynb` file.  

### Preprocessing
Next, we preprocessed the dataset. During this phase, we handled the missing values within the dataset, removed duplicate rows, excluded certain features based on their variance and addressed the outliers in each feature.

Following that, we split the dataset into 3 sets: a training set, a validation set and a test set, and we performed some further data cleaning to the training set to aid the machine learning models to generalize better the data and achieve better results. The code used for this phase of the development is contained in the `pre-processing.ipynb` file.

### Development
Lastly, we developed the cervical cancer prediction system based on the knowledge we gained through the previous two phases. The code for the final system is contained in the `prediction system.ipynb` file.

## Other Information

You can recreate the environment using the following command:

```
conda env create -f environment.yml
```

## Authors

- [@JYe9](https://github.com/JYe9)
- [@ChristinaSarogl](https://github.com/ChristinaSarogl)