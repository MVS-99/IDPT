# Author -> MVS - Manuel Vallejo Sabadell (MVS-99 github)
#=================== 00 ===========================#
# Importing libraries to be used in the microassignment.
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm

import warnings

warnings.filterwarnings("ignore")

#=================== 01 ===========================#
# Import the CSV data set into a pandas dataframe.
df_production = pd.read_csv('production.csv')

#=================== 02 ===========================#
# Use SVM with different kernels to fit the data
# The very first chore would be to separate the class from the data

df_feat = df_production[['input','output','time']].values
df_targ = df_production['efficient'].values

# Once separated, we have the option to perform an analysis
# with train/test data, but as not instructed, this code analyses
# the data as a whole without testing first

# SVM model array for linear, polynomial, Radial Basis Function kernal
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
classifiers = []
for kernel in kernels:
    clf = svm.SVC(kernel=kernel)
    clf.fit(df_feat, df_targ)
    classifiers.append(clf)

# Create a grid of points for plotting the decision boundary
xx1, yy1, zz1 = np.meshgrid(np.linspace(7, 24, 50), np.linspace(7, 24, 50), [15])

z_min_o, z_max_o = np.min([clf.decision_function(np.c_[xx1.ravel(), yy1.ravel(), zz1.ravel()]) for clf in classifiers]), np.max([clf.decision_function(np.c_[xx1.ravel(), yy1.ravel(), zz1.ravel()]) for clf in classifiers])

# Plot the decision boundaries for the different classifiers in predict mode
fig, ax = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle('SVM with Different Kernels\n(prediction)', fontsize=16, y = 0.95)
for i, kernel in enumerate(kernels):
    clf = classifiers[i]
    Z = clf.predict(np.c_[xx1.ravel(), yy1.ravel(), zz1.ravel()])
    Z = Z.reshape(xx1.shape[:-1])  # Remove the last dimension
    # Plot the decision boundary
    countour = ax[i//2, i%2].contourf(xx1[:,:,0], yy1[:,:,0], Z, levels=np.linspace(0, 1, 11), cmap=plt.cm.Paired, alpha=0.8)
    # Plot the original data points
    scatter = ax[i//2, i%2].scatter(df_feat[:, 0], df_feat[:, 1], c=df_targ, cmap=plt.cm.Paired, edgecolors='k')
    ax[i//2, i%2].set_xlim(xx1.min(), xx1.max())
    ax[i//2, i%2].set_ylim(yy1.min(), yy1.max())
    ax[i//2, i%2].set_xlabel('Input')
    ax[i//2, i%2].set_ylabel('Output')
    ax[i//2, i%2].set_title(f'Kernel: {kernel.capitalize()}')

# Create a legend for the scatter plot
fig.legend(*scatter.legend_elements(), loc="right", title="Efficient", bbox_to_anchor=(1, 0.5), borderaxespad=0.)
cbar = plt.colorbar(countour, ax=ax[:,0], location='left')
cbar.ax.set_ylabel('Efficiency', rotation=90, labelpad=15)
# As we can see, sigmoid its not the best option to represent the data
plt.savefig('SVM_without_overfitting_predict.png')

# Plot the decision boundaries for the different classifiers in decision_function mode
fig, ax = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle('SVM with Different Kernels\n(decision function)', fontsize=16, y = 0.95)
for i, kernel in enumerate(kernels):
    clf = classifiers[i]
    Z = clf.decision_function(np.c_[xx1.ravel(), yy1.ravel(), zz1.ravel()])
    Z = Z.reshape(xx1.shape[:-1])  # Remove the last dimension
    # Plot the decision boundary
    countour = ax[i//2, i%2].contourf(xx1[:,:,0], yy1[:,:,0], Z, levels=np.linspace(z_min_o, z_max_o, 11), cmap=plt.cm.Paired, alpha=0.8)
    # Plot the original data points
    scatter = ax[i//2, i%2].scatter(df_feat[:, 0], df_feat[:, 1], c=df_targ, cmap=plt.cm.Paired, edgecolors='k')
    ax[i//2, i%2].set_xlim(xx1.min(), xx1.max())
    ax[i//2, i%2].set_ylim(yy1.min(), yy1.max())
    ax[i//2, i%2].set_xlabel('Input')
    ax[i//2, i%2].set_ylabel('Output')
    ax[i//2, i%2].set_title(f'Kernel: {kernel.capitalize()}')

# Create a legend for the scatter plot
fig.legend(*scatter.legend_elements(), loc="lower right", title="Efficient", bbox_to_anchor=(1, 0.5), borderaxespad=0.)
cbar = plt.colorbar(countour, ax=ax[:,0], location='left')
cbar.ax.set_ylabel('Efficiency', rotation=90, labelpad=15)
# As we can see, sigmoid its not the best option to represent the data
plt.savefig('SVM_without_overfitting_decision.png')

# Define a function to perturb the input and output values of some data points
def perturb_data(df_feat, df_targ, num_samples=2, perturb_factor=0.2):
    # Choose num_samples random indices
    indices = np.random.choice(df_feat.shape[0], num_samples, replace=False)
    # Perturb the input and output values of the chosen data points
    df_feat_perturbed = df_feat.copy().astype('float64')
    df_targ_perturbed = df_targ.copy().astype('float64')
    for index in indices:
        df_feat_perturbed[index] += perturb_factor * np.random.randn(df_feat.shape[1])
        df_targ_perturbed[index] = 1 - df_targ_perturbed[index]
    # Stack the perturbed data to the original data set
    df_feat_overlapping = np.vstack((df_feat, df_feat_perturbed))
    df_targ_overlapping = np.hstack((df_targ, df_targ_perturbed))
    return df_feat_overlapping, df_targ_overlapping

# Generate 2 overlapping samples based on the original data set with a perturb factor of 0.1
df_feat_overlapping, df_targ_overlapping = perturb_data(df_feat, df_targ, num_samples=2, perturb_factor=0.2)

# Use SVM with different kernels to fit the data with overlapping samples
classifiers_overlapping = []
for kernel in kernels:
    clf = svm.SVC(kernel=kernel)
    clf.fit(df_feat_overlapping, df_targ_overlapping)
    classifiers_overlapping.append(clf)

z_min, z_max = np.min([clf.decision_function(np.c_[xx1.ravel(), yy1.ravel(), zz1.ravel()]) for clf in classifiers_overlapping]), np.max([clf.decision_function(np.c_[xx1.ravel(), yy1.ravel(), zz1.ravel()]) for clf in classifiers_overlapping])

# Plot the decision boundaries for the different classifiers
fig, ax = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle('SVM with Different Kernels & Overlapping samples\n(predict)', fontsize=16, y = 0.935)
for i, kernel in enumerate(kernels):
    clf = classifiers_overlapping[i]
    Z = clf.predict(np.c_[xx1.ravel(), yy1.ravel(), zz1.ravel()])
    Z = Z.reshape(xx1.shape[:-1])  # Remove the last dimension
    # Plot the decision boundary
    countour = ax[i//2, i%2].contourf(xx1[:,:,0], yy1[:,:,0], Z, levels=np.linspace(0, 1, 11), cmap=plt.cm.Paired, alpha=0.8)
    # Plot the original data points
    scatter = ax[i//2, i%2].scatter(df_feat_overlapping[:, 0], df_feat_overlapping[:, 1], c=df_targ_overlapping, cmap=plt.cm.Paired, edgecolors='k')
    ax[i//2, i%2].set_xlim(xx1.min(), xx1.max())
    ax[i//2, i%2].set_ylim(yy1.min(), yy1.max())
    ax[i//2, i%2].set_xlabel('Input')
    ax[i//2, i%2].set_ylabel('Output')
    ax[i//2, i%2].set_title(f'Kernel: {kernel.capitalize()}')

# Create a legend for the scatter plot
fig.legend(*scatter.legend_elements(), loc="lower right", title="Efficient", bbox_to_anchor=(1, 0.5), borderaxespad=0.)
cbar = plt.colorbar(countour, ax=ax[:,0], location='left')
cbar.ax.set_ylabel('Efficiency', rotation=90, labelpad=15)
plt.savefig('SVM_with_overfitting_predict.png')

# Plot the decision boundaries for the different classifiers
fig, ax = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle('SVM with Different Kernels & Overlapping samples\n(decision function)', fontsize=16, y = 0.935)
for i, kernel in enumerate(kernels):
    clf = classifiers_overlapping[i]
    Z = clf.decision_function(np.c_[xx1.ravel(), yy1.ravel(), zz1.ravel()])
    Z = Z.reshape(xx1.shape[:-1])  # Remove the last dimension
    # Plot the decision boundary
    countour = ax[i//2, i%2].contourf(xx1[:,:,0], yy1[:,:,0], Z, levels=np.linspace(z_min, z_max, 11), cmap=plt.cm.Paired, alpha=0.8)
    # Plot the original data points
    scatter = ax[i//2, i%2].scatter(df_feat_overlapping[:, 0], df_feat_overlapping[:, 1], c=df_targ_overlapping, cmap=plt.cm.Paired, edgecolors='k')
    ax[i//2, i%2].set_xlim(xx1.min(), xx1.max())
    ax[i//2, i%2].set_ylim(yy1.min(), yy1.max())
    ax[i//2, i%2].set_xlabel('Input')
    ax[i//2, i%2].set_ylabel('Output')
    ax[i//2, i%2].set_title(f'Kernel: {kernel.capitalize()}')

# Create a legend for the scatter plot
fig.legend(*scatter.legend_elements(), loc="lower right", title="Efficient", bbox_to_anchor=(1, 0.5), borderaxespad=0.)
cbar = plt.colorbar(countour, ax=ax[:,0], location='left')
cbar.ax.set_ylabel('Efficiency', rotation=90, labelpad=15)
plt.savefig('SVM_with_overfitting_decision.png')