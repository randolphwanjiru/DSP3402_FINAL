
![UTALOGO](https://github.com/randolphwanjiru/DSP3402/assets/107207718/7c99a2e5-fd3b-4572-9ec4-467c24b5030b)

# Natural Language Processing with Disaster Tweets
This repository holds an attempt to apply NLP techniques to tweets to predict which tweets were about real disaters. the data can be found at the NLP with diaster challenge tweets
https://www.kaggle.com/competitions/nlp-getting-started/overview 

## Overview

The objective involved categorizing tweets into two classes: those related to actual disasters and those unrelated. This task was part of a classification challenge, requiring the utilization of textual data from tweets. The goal was to accurately identify and differentiate between disaster-related and non-disaster-related tweets.
The approach undertaken for this task encompassed the application of various Natural Language Processing (NLP) techniques. These techniques involved preprocessing and analyzing the textual content of tweets. Different machine learning models were employed in a classification setting to discern and classify tweets accurately. The models were trained and evaluated on their ability to correctly categorize tweets into the designated classes.
Among the various models tested, the super vector machine model exhibited the highest performance, achieving an F1 score of 0.7698. Subsequently, when submitted to the Kaggle platform, the model yielded a score of 79.74%. This score represents the accuracy or predictive performance of the model compared to the ground truth data.

## Summary of Workdone

Include only the sections that are relevant an appropriate.

### Data

* Data:
  * Type: Textual Data
    * Input: medical images (1000x1000 pixel jpegs), CSV file: image filename -> diagnosis
    * Input: CSV file of features, output: signal/background flag in 1st column.
  * Size: 1.43MB
  * Instances (Train, Test, Validation Split): how many data points? Ex: 1000 patients for training, 200 for testing, none for validation

#### Preprocessing / Clean up

* Describe any manipulations you performed to the data.

#### Data Visualization

Show a few visualization of the data and say a few words about what you see.

### Problem Formulation

* Define:
  * Input / Output
  * Models
    * Describe the different models you tried and why.
  * Loss, Optimizer, other Hyperparameters.

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

* In this section, provide instructions at least one of the following:
   * Reproduce your results fully, including training.
   * Apply this package to other data. For example, how to use the model you trained.
   * Use this package to perform their own study.
* Also describe what resources to use for this package, if appropirate. For example, point them to Collab and TPUs.

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

Scikit-learn  

numpy  

matplotlib  

seaborn  

nltk (natural language processing tool kit)  

spaCy
### Data

* Point to where they can download the data.
* Lead them through preprocessing steps, if necessary.

### Training

* Describe how to train the model

#### Performance Evaluation

* Describe how to run the performance evaluation.


## Citations

* Provide any references.







