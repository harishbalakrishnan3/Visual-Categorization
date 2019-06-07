"""
A sample python script that illustrates how to use the prototype module.
As a first step, we need to find the model's parameters - c,w (we will assume r = 1).
This is done using MLE. After we find the parameters, we use them to find the corresponding probabilities using the
functions from the prototype module.
The following script illustrates the procedure to be followed. Prototype model predicted probabilities for all 16
stimuli are derived and the graph is plotted in the end.

References : Context Theory of Classification Learning , Medin and Shaffer, 1978

"""


import numpy as np, pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from prototype import *


#######################################################################################################################
# Setting up the data
#######################################################################################################################

data = pd.read_csv('../../datasets/medin1978/stimuli.csv', sep=",")
transfer_stimuli = data.values
data = pd.read_csv('../../datasets/medin1978/observed_probabilities.csv', sep=",")
observed_transfer = list(data.values[:,0])

# Training happens with only 9 stimuli
stimuli = transfer_stimuli[0:9][:]
observed = observed_transfer[:9]


categories = [[0, 1, 2, 3, 4, 9, 10, 11, 12], [5, 6, 7, 8, 13, 14, 15]]
prototypes = [[1, 1, 1, 1], [0, 0, 0, 0]]

#######################################################################################################################
# Defining functions
#######################################################################################################################


def RMSD(observed, predicted):
    total_stimuli = len(observed)
    s = 0
    for i in range(total_stimuli):
        s += (observed[i]-predicted[i])**2
    return (s/total_stimuli)**0.5


def objective(params):
    c, w = params[0], params[1:]
    total_stimuli = np.shape(stimuli)[0]
    #total_dimensions = np.shape(stimuli)[1]
    predicted = []
    for i in range(total_stimuli):
        temp = probability_of_category(categories, 0, prototypes, w, stimuli, i, r, c)
        predicted.append(temp)
    return RMSD(observed,predicted)


def weight_constraint(params):
    weights = params[1:]
    num_dimensions = len(weights)
    s = 1
    for i in range(num_dimensions):
        s -= weights[i]
    return s

#######################################################################################################################
# Performing MLE to find the ideal c and w that minimizes the RMSD between observed and predicted probabilities
#######################################################################################################################

con1 = {'type': 'eq', 'fun': weight_constraint}
cons = [con1]

# Define Bounds for c and w
bounds = []
cBound = (1, np.inf)
bounds.append(cBound)
numWeights = total_dimensions = np.shape(stimuli)[1]
for i in range(numWeights):
    bounds.append((0, 1))
bnds = tuple(bounds)

# Guess Parameters
c = 1
w = [0.25, 0.25, 0.25, 0.25]
r = 1

# Define the parameters for the objective function - c and w
params = []
params.append(c)
params += w

# MLE
results = minimize(objective, params, method='SLSQP', bounds=bnds, constraints=cons, options={'disp': True})
c, w = results.x[0], results.x[1:]

total_stimuli_transfer = np.shape(transfer_stimuli)[0]
p = []

for i in range(total_stimuli_transfer):
    p.append(probability_of_category(categories, 0, prototypes, w, transfer_stimuli, i, r, c))

print('\n\n')
print(c, w)

#######################################################################################################################
# Plotting Graphs
#######################################################################################################################

plt.style.use('ggplot')
fig= plt.figure(figsize=(8,7))
x = [x for x in range(1, 17)]
plt.plot(x, observed_transfer, '-p', alpha=0.75, label="Observed")
plt.plot(x, p, '-p', alpha=0.75, label="Predicted")
plt.xlabel('Stimulus ID #')
plt.ylabel('Probability of categorizing the stimulus into category A')
plt.title('Observed v/s Prototype Model predicted probabilities')
plt.legend(loc='upper left', frameon=True)
plt.savefig('../../datasets/medin1978/figure1.png')
plt.show()
