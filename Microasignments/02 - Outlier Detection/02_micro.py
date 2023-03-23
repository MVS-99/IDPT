# Author -> MVS - Manuel Vallejo Sabadell (MVS-99 github)
#=================== 00 ===========================#
# Importing libraries to be used in the microassignment.
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope

#=================== 01 ===========================#
# Import the CSV data set into a pandas dataframe.
df_machine = pd.read_csv('failure_in_one_month.csv')

#=================== 02 ===========================#
# First, we need to create an vector of LocalOutlierFactor models in order
# to compare the effect different values of neighbors has in the data
models_LOF = []
outliers_score = []
for n_neighbors in [3,5,8,12,16]:
    models_LOF.append(LocalOutlierFactor(n_neighbors = n_neighbors))
    outliers_score.append(models_LOF[-1].fit_predict(df_machine))

#=================== 03 ===========================#
# Once tried mutiple different models, it is instructed to select the one
# with 5 neighbors to represent using the fit_predict function which was 
# obtained previously
code_colors = np.array(['#377eb8', '#ff7f00'])
plt.figure(1)
plt.scatter(df_machine.values[:, 0], df_machine.values[:, 1], 
            color=code_colors[(outliers_score[1]+1)//2])
plt.title('Outlier detection in machine data')
plt.xlabel("Machine Temperature")
plt.ylabel("Average Machine Load")
plt.savefig('scatter_outlier_machine.png')
#We can see four outliers which are visually recognizable

#=================== 04 & 05 ===========================#
# In order to use the EllipticEnvelope algorithm, we have to considere some 
# contamination values, which are typically in the range of [0.01,0.1] and 
# represent the expected proportion of outliers within the dataset.
contamination_values = [0.01,0.025,0.05,0.075,0.1]
fit_EE = []
outliers_decision = []
outliers = []
i = 2
for cont_control in contamination_values:
    model_EE = EllipticEnvelope(contamination = cont_control)
    fit_EE.append(model_EE.fit(df_machine))
    # Now that we fitted the EllipticEnvelope to our data, the decision
    # function method (which returns a score for each observation based on
    # the Mahalanobis distance - distance between point and distribution
    # taking into account covariance) to the center of the elliptical envelope
    outliers_decision.append(fit_EE[-1].decision_function(df_machine))
    # As no particular contamination value was instructed to plot, all of them
    # will be tested:
    plt.figure(i)
    plt.plot(outliers_decision[i-2])
    plt.xlabel('Observation Index')
    plt.xticks(np.arange(0, 16, 1))
    plt.yticks(np.arange(-400,1150,150))
    plt.ylabel('Decision Function Score')
    plt.title('Decision Function Plot - Case ' + str(i-2))
    plt.savefig('decision_function_plot_{' + str(i-1) + '}.png')
    i += 1
    outliers.append(np.where(outliers_decision[-1] < 0)[0])
# The decision function plot can be better visualized if we mix them alltogether
plt.figure(i+1)
plt.plot(outliers_decision[0], label='0.01 Contamination')
plt.plot(outliers_decision[1], label='0.025 Contamination')
plt.plot(outliers_decision[2], label='0.05 Contamination')
plt.plot(outliers_decision[3], label='0.075 Contamination')
plt.plot(outliers_decision[4], label='0.1 Contamination')
plt.legend()
plt.xlabel('Observation Index')
plt.xticks(np.arange(0, 16, 1))
plt.ylabel('Decision Function Score')
plt.title('Decision Function Plot - All cases')
plt.savefig('decision_function_plot_{all}.png')
# Now, remember that decision_function = score_samples - offset_ (an attribute
# of the Elliptic Envelope model). Negative samples are likely outliers. 
# Considering the outliers list obtained before, we can see that both the
# 14th and 15th (although the majority, the less contaminated ones, just
# include the 15th value as an outlier)
print('The outlier is the 15th [' + str(df_machine.values[15,0]) + ',' + 
      str(df_machine.values[15,1]) + '] for the first three contamination values contemplated.\n')
print('In case we take the most restrictive option (or the highest contamination values) the 14th [' + 
      str(df_machine.values[14,0]) + ',' + str(df_machine.values[14,1]) + '] is also an outlier.\n' )

print('This implies, therefore, that probably only the last two observations are outliers')


