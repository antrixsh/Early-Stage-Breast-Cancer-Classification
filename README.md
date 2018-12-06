# Early-Stage-Breast-Cancer-Classification
Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image. n the 3-dimensional space is that described in: [K. P. Bennett and O. L. Mangasarian: "Robust Linear Programming Discrimination of Two Linearly Inseparable Sets", Optimization Methods and Software 1, 1992, 23-34].  This database is also available through the UW CS ftp server: ftp ftp.cs.wisc.edu cd math-prog/cpo-dataset/machine-learn/WDBC/  Also can be found on UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29
# Attribute Information:

1) ID number 2) Diagnosis (M = malignant, B = benign) 3-32)

Ten real-valued features are computed for each cell nucleus:

a) radius (mean of distances from center to points on the perimeter) b) texture (standard deviation of gray-scale values) c) perimeter d) area e) smoothness (local variation in radius lengths) f) compactness (perimeter^2 / area - 1.0) g) concavity (severity of concave portions of the contour) h) concave points (number of concave portions of the contour) i) symmetry j) fractal dimension ("coastline approximation" - 1)

The mean, standard error and "worst" or largest (mean of the three largest values) of these features were computed for each image, resulting in 30 features. For instance, field 3 is Mean Radius, field 13 is Radius SE, field 23 is Worst Radius.

All feature values are recoded with four significant digits.

Missing attribute values: none

Class distribution: 357 benign, 212 malignant
# STEP #1: PROBLEM STATEMENT

    Predicting if the cancer diagnosis is benign or malignant based on several observations/features

    30 features are used, examples:

      - radius (mean of distances from center to points on the perimeter)
      - texture (standard deviation of gray-scale values)
      - perimeter
      - area
      - smoothness (local variation in radius lengths)
      - compactness (perimeter^2 / area - 1.0)
      - concavity (severity of concave portions of the contour)
      - concave points (number of concave portions of the contour)
      - symmetry 
      - fractal dimension ("coastline approximation" - 1)

    Datasets are linearly separable using all 30 input features
    Number of Instances: 569
    Class Distribution: 212 Malignant, 357 Benign
    Target class:

       - Malignant
       - Benign

https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
# STEP #2: IMPORTING DATA
  #import libraries 
  import pandas as pd # Import Pandas for data manipulation using dataframes
  import numpy as np # Import Numpy for data statistical analysis 
  import matplotlib.pyplot as plt # Import matplotlib for data visualisation
  import seaborn as sns # Statistical data visualization
  #%matplotlib inline
  #Import Cancer data drom the Sklearn library
  from sklearn.datasets import load_breast_cancer
  cancer = load_breast_cancer()
# STEP #3: VISUALIZING THE DATA
  ![github-small](https://github.com/antrixsh/Early-Stage-Breast-Cancer-Classification/blob/master/Data_Visualization/fig1.PNG)
  ![github-small](https://github.com/antrixsh/Early-Stage-Breast-Cancer-Classification/blob/master/Data_Visualization/fig2.PNG)
  ![github-small](https://github.com/antrixsh/Early-Stage-Breast-Cancer-Classification/blob/master/Data_Visualization/fig3.PNG)
  ![github-small](https://github.com/antrixsh/Early-Stage-Breast-Cancer-Classification/blob/master/Data_Visualization/fig4.PNG)
# STEP #4: MODEL TRAINING (FINDING A PROBLEM SOLUTION)
  Refer Code Notebook
# STEP #5: EVALUATING THE MODEL
  Refer Code Notebook
# STEP #6: IMPROVING THE MODEL
  Refer Code Notebook  
# IMPROVING THE MODEL - PART 2
  param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']} 
  from sklearn.model_selection import GridSearchCV
  grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=4)
  grid.fit(X_train_scaled,y_train)
