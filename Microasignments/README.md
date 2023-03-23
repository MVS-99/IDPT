# MicroAssignments

The following subfolders define the different so-called MicroAssignments. They are, in words of the professor ([Mario Muñoz Organero](https://www.linkedin.com/in/mario-mu%C3%B1oz-3549697/))
> “Micro” assignments to solve “small” challenges related to the theoretical sessions.
And they have the objective of consolidate the theoretical algorithm knowledge with practical use cases.

There are five different assignments throught the course:

## 01 - Fertility Analysis (Basic understanding/manipulation of data structures)

This code is an analysis of a [fertility dataset](https://datahub.io/machine-learning/fertility) which consists of 100 instances with 10 attributes. Each attribute represents different characteristics of patients. These attributes include:

- **Season (1)**: season in which the analysis was performed *(winter, spring, summer, fall)*
- **Age (2)**: age at the time of analysis *(18-36)*
- **Childish disease (3)**: presence of childhood diseases, e.g. chickenpox, measles, mumps, polio *(yes, no)*
- **Trauma (4)**: presence of accidents or serious trauma *(yes, no)*
- **Surgical intervention (5)**: previous surgical intervention *(yes, no)*
- **High fevers in the last year (6)**: occurrence of high fevers in the last year *(less than three months ago, more than three months ago, no)*
- **Frequency of alcohol consumption (7)**: frequency of alcohol intake *(several times a day, every day, several times a week, once a week, hardly ever or never)*
- **Smoking habit (8)**: smoking habit *(never, occasional, daily)*
- **Number of hours spent sitting per day (9)**: number of hours spent sitting per day *(0-16)*
- **Diagnosis (10)**: diagnosis normal *(N)*, altered *(O)*

The dataset, stored in a CSV file, is stored in divided into two separate DataFrames filtered by the two possible diagnoses: normal (N) and altered (O). It computes the mean age of the patients for each diagnosis and prints a message indicating whether the mean age is higher for normal or altered diagnoses or if they are the same.

Afterward, a histogram is generated to compare the age distribution of both diagnoses. The number of bins is selected using the rule of thumb of approximately the square root of the number of rows to the highest. Readability options are implemented.

Finally, the code converts the age columns in both DataFrames to NumPy arrays for representation ease. The arrays are defined using the .values() function.

This code serves as an example of how to read and analyze data stored in a CSV file using Pandas, as well as how to generate basic visualizations using Matplotlib. It also showcases the usefulness of NumPy arrays in data analysis

## 02 - Basic Outlier detection

The code is a data analysis script that uses two different outlier detection techniques to identify anomalous data points in a machine failure dataset. The two techniques used are Local Outlier Factor (LOF) and Elliptic Envelope (EE).

The LOF algorithm is a type of density-based outlier detection technique. It measures the local density of a point relative to its k-nearest neighbors and identifies outliers as points that have a much lower density than their neighbors. In this script, the LOF algorithm is applied with different values of k (i.e., number of neighbors) to determine the optimal value for this parameter. The resulting outlier scores are then plotted to visualize the outlier points.

The EE algorithm, on the other hand, is a type of model-based outlier detection technique. It assumes that the data is normally distributed and fits an elliptical envelope to the data using the Mahalanobis distance metric. Outliers are identified as points that fall outside of the envelope. In this script, the EE algorithm is applied with different levels of contamination (i.e., expected proportion of outliers in the dataset) to determine the optimal level for this parameter. The resulting decision function scores are then plotted to visualize the outlier points.

The script first imports the necessary libraries for data analysis, including Pandas for data manipulation, Matplotlib for data visualization, NumPy for numerical computing, and Scikit-learn for machine learning.

Then, the machine failure dataset is imported from a CSV file using the Pandas library.

The LOF algorithm is then applied with different values of k (3, 5, 8, 12, and 16) to determine the optimal value for this parameter. The resulting outlier scores are stored in a list and plotted using a scatter plot to visualize the outlier points.

Next, the EE algorithm is applied with different levels of contamination (0.01, 0.025, 0.05, 0.075, and 0.1) to determine the optimal level for this parameter. The resulting decision function scores are stored in a list and plotted using line plots to visualize the outlier points. The decision function scores are calculated using the Mahalanobis distance metric and the offset attribute of the EE model. Negative scores indicate outliers.

Finally, the outlier points are identified based on the decision function scores and the original dataset. The results show that the last two observations in the dataset are likely outliers.

Overall, the script provides a comprehensive analysis of the machine failure dataset using two different outlier detection techniques. The LOF and EE algorithms can be used to identify anomalous data points in various datasets, which is useful for data cleaning, anomaly detection, and predictive modeling.

## 03 - Support Vector Machines for outlier classification

This is a Python script that uses Support Vector Machines (SVM) with different kernels to fit a data set.

Support Vector Machines (SVMs) are a type of supervised learning algorithm used for classification and regression analysis. In classification tasks, SVMs try to find a hyperplane that separates two classes of data points in the feature space. SVMs work by finding the optimal margin between two classes of data points. The margin is defined as the distance between the hyperplane and the closest data points of each class. The optimal hyperplane is the one that maximizes this margin.

The script reads a CSV data set (named 'production.csv') into a pandas dataframe. The data consists of three features ('input', 'output', and 'time') which are the measures obtained directly from the system and a binary classification target variable ('efficient'). The purpose is to determine if the process has been efficient or not.

The code then separates the features and target variables, and performs an analysis without testing first. It creates a grid of points for plotting the decision boundary and trains SVM models. SVMs can use different kernel functions to transform the input data into a higher-dimensional feature space, where the optimal hyperplane can be found. In this assignment the most common ones are used: linear, polynomial, radial basis function, and sigmoid.

- `Linear Kernel` : The linear kernel is the simplest kernel function, and it works well when the data points are linearly separable in the original feature space. It is defined as the dot product between the input vectors.

- `Polynomial Kernel` : The polynomial kernel is a non-linear kernel function that maps the input data into a higher-dimensional feature space using polynomial functions. It can handle data points that are not linearly separable in the original feature space. In this assignment, the degree of the polynomial has been left as default (3-degree polynomial).

- `Radial Basis Function (RBF) Kernel` : The RBF kernel is a popular non-linear kernel function that maps the input data into an infinite-dimensional feature space using a Gaussian function. It is widely used for classification tasks because it can handle complex decision boundaries.

- `Sigmoid Kernel` : The sigmoid kernel is a non-linear kernel function that maps the input data into a higher-dimensional feature space using sigmoid functions. It is typically used for neural networks, but it can also be used for SVMs

It then creates two plots, one with decision boundaries for different classifiers in prediction mode, and another with decision boundaries for different classifiers in decision_function mode. The plots show the decision boundaries and the original data points.

## 04 - Neural Networks

## 05 - Deep Learning

