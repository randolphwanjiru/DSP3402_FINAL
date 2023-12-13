
![UTALOGO](https://github.com/randolphwanjiru/DSP3402/assets/107207718/7c99a2e5-fd3b-4572-9ec4-467c24b5030b)

# Natural Language Processing with Disaster Tweets
This repository holds an attempt to apply NLP techniques to tweets to predict which tweets were about real disasters. The data can be found at the NLP with disaster tweets challenge 
https://www.kaggle.com/competitions/nlp-getting-started/overview 

## Overview

The objective involved categorizing tweets into two classes: those related to actual disasters and those unrelated. This task was part of a classification challenge, requiring the utilization of textual data from tweets. The goal was to accurately identify and differentiate between disaster-related and non-disaster-related tweets.
The approach undertaken for this task encompassed the application of various Natural Language Processing (NLP) techniques. These techniques involved preprocessing and analyzing the textual content of tweets. Different machine learning models were employed in a classification setting to discern and classify tweets accurately. The models were trained and evaluated on their ability to correctly categorize tweets into the designated classes.
Among the various models tested, the super vector machine model exhibited the highest performance, achieving an F1 score of 0.7698. Subsequently, when submitted to the Kaggle platform, the model yielded a score of 79.74%. This score represents the accuracy or predictive performance of the model compared to the ground truth data.

## Summary of Work done


### Data

* Data:
  * Type: Textual Data, CSV file
  * Size: 1.43MB
  * train data instances: 7613
  * test data instances: 3263

#### Preprocessing / Clean up

Utilized  a TF-IDF Vectorizer (TfidfVectorizer) from sci-kit-learn. The TF-IDF Vectorizer, used in the code, transformed textual data by assigning numerical values to words based on their significance in individual tweets and across the dataset. This allowed the machine learning models to understand and classify tweets regarding real disasters more effectively by focusing on relevant terms and ignoring common words. The vectorizer's parameter was set to 5000, limiting the vocabulary size to prioritize the most important words, and enhancing the models' classification accuracy by considering the contextual relevance of specific terms within the tweets.

#### Data Visualization
![download](https://github.com/randolphwanjiru/DSP3402_FINAL/assets/107207718/80b5ffd5-28df-4929-9c06-bf51c35d3341)  
57 percent of the data set consists of non-diaster tweets. 43% percents accounts for disaster-related tweeets   
![download](https://github.com/randolphwanjiru/DSP3402_FINAL/assets/107207718/a49654c3-58b1-4e82-9627-f48ad3a733ce)
The SVM machine model is the best-performing model based on the metrics 
### Problem Formulation

Inputs:  
* Training Data: Contains columns like 'id', 'keyword', 'location', 'text', 'target' loaded from 'train.csv'.
* Testing Data: Structured similarly to training data but lacks the 'target' column, loaded from 'test.csv'.

 
Outputs:  
* Performance Metrics: Includes F1 Score, Accuracy, Precision, and Recall for each model on the validation set.
* Best Model Information: Displays details of the highest-performing model (name, F1 Score, Accuracy, Precision, Recall).  



* Multinomial Naive Bayes (MultinomialNB):  
Chosen for its efficiency in handling text data and computational simplicity.
* Logistic Regression:  
Employed as a basic linear model for comparison and its effectiveness in binary classification.
* Support Vector Machine (SVM):  
Selected due to its adaptability to various data types and suitability for high-dimensional spaces.
* Random Forest:  
Utilized as an ensemble method capable of managing complex data relationships and minimizing overfitting.
* MLPClassifier (Multi-layer Perceptron):  
A flexible neural network capable of learning intricate patterns in data.
### Training

                                  * Describe the training:
                               * How you trained: software and hardware.
                       * How did training take.
                        * Training curves (loss vs epoch for test/train).
                            * How did you decide to stop training.
                                  * Any difficulties? How did you resolve them?

### Performance Comparison

* Clearly define the key performance metric(s). Multonmoal Naive Bayes F1 Score: 0.7463
* Show/compare results in one table.            Logistic regression score: 0.7494
* ![image](https://github.com/randolphwanjiru/DSP3402_FINAL/assets/107207718/abcfa793-e1f8-44b7-bfb8-d9f366aa7a0b)

* Show one (or few) visualization(s) of results, for example ROC curves.

### Conclusions
Logistic Regression slightly outperformed Multinomial Naive Bayes based on the F1 Score

### Future Work

* What would be the next thing that you would try.
* What are some other studies that can be done starting from here.

## How to reproduce results
To reproduce the results or utilize the trained model, follow these steps:  
Environment Setup:  

* Ensure Python and required libraries (e.g., pandas, scikit-learn) are installed in your environment. Use virtual environments like Anaconda or pip to manage dependencies.
Data Loading and Preprocessing:

* Load the dataset ('train.csv' and 'test.csv') into your working directory using pandas.
Split the data into features ('text') and target labels ('target').
Training the Support Vector Machine (SVM) Model:

* Use scikit-learn's SVM implementation ('sklearn.svm.SVC') and preprocess text data using TF-IDF Vectorization (as previously implemented).
Train the SVM model on the training data.
Model Evaluation:

Evaluate the model's performance using metrics like F1 Score, Accuracy, Precision, and Recall on a validation set.

### Overview of files in repository

* Describe the directory structure, if any.
* List all relavent files and describe their role in the package.
* An example:
  * utils.py: various functions that are used in cleaning and visualizing data.
  * preprocess.ipynb: Takes input data in CSV and writes out data frame after cleanup.
  * visualization.ipynb: Creates various visualizations of the data.
  * models.py: Contains functions that build the various models.
  * training-model-1.ipynb: Trains the first model and saves model during training.
  * training-model-2.ipynb: Trains the second model and saves model during training.
  * training-model-3.ipynb: Trains the third model and saves model during training.
  * performance.ipynb: loads multiple trained models and compares results.
  * inference.ipynb: loads a trained model and applies it to test data to create kaggle submission.

* Note that all of these notebooks should contain enough text for someone to understand what is happening.

### Software Setup

### Data

* Point to where they can download the data.
* Lead them through preprocessing steps, if necessary.

### Training

* Describe how to train the model

#### Performance Evaluation
* Data Preparation: Loads and splits the dataset into training and validation sets.  

* Model Definition: Initializes models like Naive Bayes, Logistic Regression, SVM, Random Forest, MLPClassifier.  

* Training and Evaluation Loop: Trains each model on the training set and evaluates on the validation set using F1 Score, Accuracy, Precision, and Recall.  

* Best Model Selection: Identifies the model with the highest F1 Score as the best-performing one.  

* Performance Metrics Display: Generates a table displaying performance metrics for all models and prints details of the best model.  

* Prediction and Submission: Uses the best model to predict on the test data and creates a submission CSV file.  

## Citations

* Provide any references.







