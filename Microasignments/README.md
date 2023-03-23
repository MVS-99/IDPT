# MicroAssignments

The following subfolders define the different so-called MicroAssignments. They are, in words of the professory (Mario Muñoz Organero)
> “Micro” assignments to solve “small” challenges related to the theoretical sessions.
And they have the objective of consolidate the theoretical algorithm knowledge with practical use cases.

There are five different assignments throught the course:

## 01 - Fertility Analysis (Basic understanding/manipulation of data structures)

## 02 - Basic Outlier detection

## 03 - Support Vector Machines for outlier classification

This is a Python script that uses Support Vector Machines (SVM) with different kernels to fit a data set.

Support Vector Machines (SVMs) are a type of supervised learning algorithm used for classification and regression analysis. In classification tasks, SVMs try to find a hyperplane that separates two classes of data points in the feature space. SVMs work by finding the optimal margin between two classes of data points. The margin is defined as the distance between the hyperplane and the closest data points of each class. The optimal hyperplane is the one that maximizes this margin.

The script reads a CSV data set (named 'production.csv') into a pandas dataframe. The data consists of three features ('input', 'output', and 'time') and a binary target variable ('efficient').

The code then separates the features and target variables, and performs an analysis without testing first. It creates a grid of points for plotting the decision boundary and trains SVM models. SVMs can use different kernel functions to transform the input data into a higher-dimensional feature space, where the optimal hyperplane can be found. In this assignment the most common ones are used: linear, polynomial, radial basis function, and sigmoid.

- `Linear Kernel` : The linear kernel is the simplest kernel function, and it works well when the data points are linearly separable in the original feature space. It is defined as the dot product between the input vectors.

- `Polynomial Kernel` : The polynomial kernel is a non-linear kernel function that maps the input data into a higher-dimensional feature space using polynomial functions. It can handle data points that are not linearly separable in the original feature space. In this assignment, the degree of the polynomial has been left as default (3-degree polynomial).

- `Radial Basis Function (RBF) Kernel` : The RBF kernel is a popular non-linear kernel function that maps the input data into an infinite-dimensional feature space using a Gaussian function. It is widely used for classification tasks because it can handle complex decision boundaries.

- `Sigmoid Kernel` : The sigmoid kernel is a non-linear kernel function that maps the input data into a higher-dimensional feature space using sigmoid functions. It is typically used for neural networks, but it can also be used for SVMs

It then creates two plots, one with decision boundaries for different classifiers in prediction mode, and another with decision boundaries for different classifiers in decision_function mode. The plots show the decision boundaries and the original data points.

## 04 - Neural Networks

## 05 - Deep Learning

