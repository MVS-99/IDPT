# Author -> MVS - Manuel Vallejo Sabadell (MVS-99 github)
#=================== 00 ===========================#
# Importing libraries to be used in the microassignment.
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

#=================== 01 ===========================#
# Import the CSV data set into a pandas dataframe.
df_moons = pd.read_csv('moons.csv')

#=================== 02 ===========================#
# Use the first 2 columns (f1 - f2) as features and 3rd as labels
df_feat = df_moons[['f1','f2']].values
df_label = df_moons[['label']].values

#=================== 03 ===========================#
# Use the train_test_split in sklearn to randomly split the data into train and test
feat_train, feat_test, label_train, label_test = train_test_split(df_feat, df_label, test_size=0.33, random_state=42)

# Train a Multi-Layer Perceptron Classifier with different parameters
neurons = [5, 10, 20, 30, 40]
layers = [1, 2, 3]

fig, axs = plt.subplots(5, 3, figsize=(10, 15))
fig.suptitle('Boundary detection for label classification', fontsize=16, y=1)

figp, axsp = plt.subplots(5, 3, figsize=(10,15))
figp.suptitle('Validation of model predictions on test data', fontsize = 16, y=1)

for i, n in enumerate(neurons):
    for j, l in enumerate(layers):
        # Code to 
        mlp = MLPClassifier(hidden_layer_sizes=(n,) * l, max_iter=1000)
        mlp.fit(feat_train, label_train)

        # Use the trained model to make predictions on the test data
        label_pred = mlp.predict(feat_test)


        # Plot the decision boundary
        ax = axs[i, j]
        ax.set_title("Neurons: {} - Layers: {}".format(n, l))
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        Xneg = feat_train[np.squeeze(label_train == 0)]
        Xpos = feat_train[np.squeeze(label_train == 1)]
        ax.scatter(Xneg[:, 0], Xneg[:, 1], color='r', label='0')
        ax.scatter(Xpos[:, 0], Xpos[:, 1], color='b', label='1')
        ax.legend()

        xx1, yy1 = np.meshgrid(np.linspace(-1, 3, 50), np.linspace(-1, 2, 50))
        Z1 = mlp.predict_proba(np.c_[xx1.ravel(), yy1.ravel()])
        Z1 = Z1[:, 0].reshape(xx1.shape)
        contour = ax.contour(xx1, yy1, Z1, levels=[0.5], linewidths=2, linestyles='dashed', colors='green')
        ax.clabel(contour, inline=1, fontsize=10)



        # Plot the prediction
        ax = axsp[i, j]
        ax.set_title("Neurons: {} - Layers: {}".format(n, l))
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        Xneg_test = feat_test[np.squeeze(label_test == 0)]
        Xpos_test = feat_test[np.squeeze(label_test == 1)]
        Xneg_pred = feat_test[np.squeeze(label_pred == 0)]
        Xpos_pred = feat_test[np.squeeze(label_pred == 1)]
        ax.scatter(Xneg_test[:, 0], Xneg_test[:, 1], color='r', label='True 0')
        ax.scatter(Xpos_test[:, 0], Xpos_test[:, 1], color='b', label='True 1')
        ax.scatter(Xneg_pred[:, 0], Xneg_pred[:, 1], marker='x', color='m', label='Pred 0')
        ax.scatter(Xpos_pred[:, 0], Xpos_pred[:, 1], marker='x', color='c', label='Pred 1')
        ax.legend(fontsize = 8)
fig.tight_layout()
figp.tight_layout()
fig.savefig('boundary_detection_label_classification.png')
figp.savefig('prediction_test_data.png')