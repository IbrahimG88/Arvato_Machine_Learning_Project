### This project is my Capstone project I worked on for the Udacity 
Machine Learning Engineer Nanodegree

The dataset is provided by Arvato financial solutions, a german company.
The project is divided into two sections. 

#### Part 1:
Creating a customer segmentation report using unsupervised learning machine learning model.

#### Part 2:
Creating a supervised model to predict whether a person providing a set of features is a possible target and high likely 
to become a client for the company.

### Project Overview
In this project datasets from Arvato financials solutions company have been provided. The main aim of the project is to find what does the customers of the company have in common with the general population in Germany, since Arvato is a German company. To answer this question unsupersived learning models were applied to define the key features for the cluster of the population that are similar in features to the current customers of the company. The second part of the project is occupied with creating a supervised model that is able to predict based on provided features whether a person is possible customer if the financial solution of Arvato was presented to those people.

For the first part of the project to be able to study the main features of the general population group and the current customers for Arvato two datasets were provided and used. A dataset for the general population and a dataset for the current Arvato customers. The same set of features is present in both datasets.

In the second part the datasets provided are: A dataset for a previous group of people that were contact using a mailout campaign. Some of them became customers, others didn’t reply or were not interested to use the company services. This was described in this dataset by a feature stating a response: a customer or not. The second dataset for this section is another group of people and their features and a prediction has to be made using the created machine learning model whether a person in this group of a bailout campaign will be a possible customer or not.


### Algorithms and techniques

The techniques used in the project for the features’ dimensionality reduction is the PCA principal component analysis to provide a smaller number of features that still cover 60% of the variance of the data to be able to use the data with supervised and unsupervised models.
For the Unsupervised part k-Means algorithm is being used to create clusters for the general population and the Arvato company customers and to be able to compare clusters of high counts of datapoint of both datasets.
In the supervised learning part an ensemble method Algorithm was used. The AdaBoost classifier which also included the DecisionTree classifier as a base estimator.
To be able to improve the quality of the Adaboost Classifier efficiency a grid search was applied to find better parameters for the classifier. Also class weight the classifier was inserted so that to give more weight distribution and higher priority for the class predicting whether a person or a datapoint is a future possible customer.

The unsupervised learning model used is the k-means. The means will use the provided data after the pca analysis with reduced dimensionality features. Then I have to decide the number of clusters that will efficient for clustering the datapoint. For this I used the elbow method which plots a curve for the distortion against the number of clusters for the data. At the elbow or where the curve starts to flat out, this area describes the best number of clusters for the means which was found to be 5 for this project. Then the kmeans creates 5 centroids and adds the nearest points to the centers to be members of each cluster. And this process repeats till the best results are reach and the model reaches convergence. The model predicts for each datapoint to which cluster this point belongs.

 The supervised learning model used is the ensemble AdaBoost model which used a number of weak learners, here these weak learners are Decision Tree classifiers as base estimators that each try to learn and predict the classes of the datapoint separately. The model uses a number of Decision Tree classifiers having certain parameters such as the max depth of the tree, stages for deciding the class of the point being worked on, the class weights which is important here as the emphasis in this model is on the class of predicted customers, here is class 1, and other parameters such as minimum number of samples per split. The number of weak of weak learners for example 50 means it will use 50 decision tree classifiers to use the result from all of these classifiers together to give weights for the data inputs and will enrich the decision by the final adaboost classifier for making decisions for the datapoint to make a prediction for which combines the knowledge gained from the 50 decision tree classifiers and based on that decides the best weight to be given to each input parameter. A gridsearch was also applied to try different parameters for the estimator to find the best results for each parameter included in the gridsearch.
