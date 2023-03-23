# Author -> MVS - Manuel Vallejo Sabadell (MVS-99 github)
#=================== 01 ===========================#
# Importing libraries to be used in the microassignment.
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#=================== 02 ===========================#
# Import the CSV data set into a pandas dataframe.
df_fert = pd.read_csv('./fertility.csv')

#=================== 03 & 04 ===========================#
# Divide into two dataframes filtered by the two possible outputs
# where N is normal diagnose and O abnormal/altered.
df_N_fert = df_fert[df_fert['output']=='N']
df_O_fert = df_fert[df_fert['output']=='O']

#=================== 05 ===========================#
m_N_age = (df_N_fert['age'].mean() * 18) + 18
m_O_age = (df_O_fert['age'].mean() * 18) + 18

if (m_N_age > m_O_age):
    print('\nMEAN AGE FOR NORMAL DIAGNOSE == ', m_N_age, '\n')
    print('MEAN AGE FOR ALTERED DIAGNOSE == ', m_O_age, '\n')
    print('The mean age for a normal diagnoses is higher than the age of a altered diagnose.\n')
elif(m_N_age < m_O_age):
    print('\nMEAN AGE FOR NORMAL DIAGNOSE == ', m_N_age, '\n')
    print('MEAN AGE FOR ALTERED DIAGNOSE == ', m_O_age, '\n')
    print('The mean age for an altered diagnose is higher than the age of a normal diagnose.\n')
else:
    print('The mean age is the same for both normal (', m_N_age,') and altered (', m_O_age, ') diagnoses.\n')

#=================== 06 & 08 ===========================#
# Rule of thumb for bins number selection --> approx(sqrt(rows)) to the highest.
print('Generating plot histogram for comparison of both outputs age distribution.\n')
plt.hist((df_N_fert["age"] * 18) + 18, bins=10, color = '#0b84a5', alpha=1, label="Normal Diagnose")
plt.hist((df_O_fert["age"] * 18) + 18, bins=4, color = '#ca472f', alpha=1, label="Altered Diagnose")
# Make the range of the ticks between the maximum and minimum values of both combined.
plt.xlim(min(((df_N_fert['age'].min())*18)+18, (df_O_fert['age'].min()*18)+18),
         max((df_N_fert['age'].max()*18)+18, (df_O_fert['age'].max()*18)+18))
plt.legend(loc="upper right")
plt.title('Age distribution comparison')
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.savefig('age_hist_comp.png')
plt.show()

#=================== 07 ===========================#
# Numpy arrays are defined as follows (.values() function).
# Also useful (if I didn´t go ahead of time and complete the plot beforehand)
# to convert them to natural age numbers for representation ease.
npa_N_fert = ((df_N_fert['age'].values * 18) + 18)
npa_O_fert = ((df_O_fert['age'].values * 18) + 18)