# Titanic Survival Predictions

In this project we will work with the Titanic dataset ( which is like a hello world for the ML ) and try to predict the passenger, who will survive the tragedy.
## Introduction
As this is a very good dataset to begin ML journey, we are going to use many algorithms and test them on our dataset.
I will be testing the following models with my training data 

-   Gaussian Naive Bayes
-   Logistic Regression
-   Support Vector Machines
-   Perceptron
-   Decision Tree Classifier
-   Random Forest Classifier
-   KNN or k-Nearest Neighbors
-   Stochastic Gradient Descent
-   Gradient Boosting Classifier

If you are not familier with any of the above algorithm, just google it, you will get a lot of tutorials there.

This repo contains:
- **main.py** : contains the python implementation of this project. 
-  **main.ipynb** : This is included for better understanding of the code. **Must** for getting the basics of every line of code.
-  **data.csv** : contains the Titanic dataset.
## Requirements:
Following python packages are required:
- numpy
- pandas
- matplotlib
- seaborn
- warnings
- sklearn

***Note -*** If you dont't have any of this package not installed on your system, you can easily install that package by running command `pip install <package-name>` or `pip3 install <package-name>`.
ex: `pip3 install wikipedia`

### Steps to run it on your system:

**Note:** I am using Ubuntu 18.04 with anaconda environment and python-3.6.8

1.  Get this project to your local system
    
2.  Change directory to current project
    
    > cd titanic_survival_prediction
    
3.  Create virtual environment _**[Optional]**_  
    Using Anaconda here( You may use python venv) 
    **Note:** Use tensorflow as backend in keras
    

-   Use the terminal or an Anaconda Prompt for the following steps:
    
    > conda create -n myenv python=3.6.8
    
-   Activate the new environment:
    
    > conda activate myenv
    

4.  Run the python file
    
    > python main.py
    
    **Note:** If you have created a virtual environment , you may leave it by running
    
    > conda deactivate

**Sample output** : After executing the command, you will be shown the accuracy of various algorithms used here for our dataset.
