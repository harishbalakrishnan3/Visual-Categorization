"""
A sample python script that illustrates GCM.
As a first step, we need to find the model's parameters - c,w,b (we assume r = 2).
This is done using MLE. After we find the parameters, we use them to find the corresponding probabilities using the
functions from the gcm module.
The following script illustrates the procedure for Subject1. GCM predicted probabilities of all four categorization
types are derived and the graph is plotted in the end.
"""

import pandas as pd, numpy as np
from scipy.optimize import minimize
from gcm import *
import matplotlib.pyplot as plt


def objective(params):
    def SSE(a, b):
        n = len(a)
        sse = 0
        for i in range(n):
            sse += (a[i] - b[i]) ** 2
        return sse

    w, c, b = params[1:numDimensions+1], params[0], params[numDimensions+1:numDimensions+1+totalCategories]
    probabilities = []
    for i in range(16):
        p = probability_of_category_J(0, stimulus_representation, w, c, r, i, categories_idx, b)
        probabilities.append(p)
    sse = SSE(probabilities,observed)
    return sse

def weightConstraint(params):
    weights = params[1:numDimensions+1]
    sum = 1
    for i in range(numDimensions):
        sum-=weights[i]
    return sum

def biasConstraint(params):
    biases = params[numDimensions+1:numDimensions+1+totalCategories]
    sum =1
    for i in range(totalCategories):
        sum-=biases[i]
    return sum

def calculateLL(probabilities):
    return -np.sum(np.log(probabilities))


data = pd.read_csv('../datasets/GCM/nosofsky1986/subject1/stimuli.csv', sep=",")
observed_df = pd.read_csv('../datasets/GCM/nosofsky1986/subject1/observed_probabilities.csv', sep=",")
stimulus_representation = data.values
numDimensions = np.shape(stimulus_representation)[1]


# Case1: Dimensional Stimuli
observed = list(observed_df.values[:, 0])
observed_dimensional = observed
categories_idx = [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]]
totalCategories = len(categories_idx)

# Guess Parameters
c= 1
w = [0.5, 0.5]
b = [0.5, 0.5]
r = 2


# Define Bounds for c, w, b
bounds = []
cBound = (0, np.inf)
bounds.append(cBound)
numWeights = 2
numBiases = 2
for i in range(numWeights):
    bounds.append((0, 1))
for i in range(numBiases):
    bounds.append((0, 1))
bnds = tuple(bounds)

# Define the parameters for the objective function - c, w, b
params = []
params.append(c)
params += w+b

# Constraints
con1 = {'type': 'eq', 'fun': weightConstraint}
con2 = {'type': 'eq', 'fun': biasConstraint}
cons = [con1, con2]

results = minimize(objective, params, method = 'SLSQP', bounds = bnds, constraints = cons, options={'disp': True})
w, c, b = results.x[1:numDimensions+1], results.x[0], results.x[numDimensions+1:numDimensions+1+totalCategories]
predicted_dimensional = calculate_probabilities(0, stimulus_representation, w, c, r, categories_idx, b)


# Case 2: Criss-Cross Stimuli
observed = list(observed_df.values[:, 1])
observed_crisscross = observed
categories_idx = [[2, 3, 6, 7, 8, 9, 12, 13], [0, 1, 4, 5, 10, 11, 14, 15]]

# Guess Parameters
c= 1
w = [0.5, 0.5]
b = [0.5, 0.5]
r = 2

# Define the parameters for the objective function - c, w, b
params = []
params.append(c)
params += w+b

results = minimize(objective, params, method = 'SLSQP', bounds = bnds, constraints = cons, options={'disp': True})
w, c, b = results.x[1:numDimensions+1], results.x[0], results.x[numDimensions+1:numDimensions+1+totalCategories]
predicted_crisscross = calculate_probabilities(0, stimulus_representation, w, c, r, categories_idx, b)


# Case 3: Interior-Exterior
observed = list(observed_df.values[:, 2])
observed_interiorexterior= observed
categories_idx = [[5, 6, 9, 10], [0, 1, 2, 3, 4, 7, 8, 11, 12, 13, 14, 15]]

# Guess Parameters
c= 1
w = [0.5, 0.5]
b = [0.5, 0.5]
r = 2

# Define the parameters for the objective function - c, w, b
params = []
params.append(c)
params += w+b

results = minimize(objective, params, method = 'SLSQP', bounds = bnds, constraints = cons, options={'disp': True})
w, c, b = results.x[1:numDimensions+1], results.x[0], results.x[numDimensions+1:numDimensions+1+totalCategories]
predicted_interiorexterior = calculate_probabilities(0, stimulus_representation, w, c, r, categories_idx, b)



# Case 4: Diagnol
observed = list(observed_df.values[:, 3])
observed_diagnol = observed
categories_idx = [[0, 1, 2, 4, 5, 8, 12], [3, 6, 7, 9, 10, 11, 13, 14, 15]]

# Guess Parameters
c= 1
w = [0.5, 0.5]
b = [0.5, 0.5]
r = 2

# Define the parameters for the objective function - c, w, b
params = []
params.append(c)
params += w+b

results = minimize(objective, params, method = 'SLSQP', bounds = bnds, constraints = cons, options={'disp': True})
w, c, b = results.x[1:numDimensions+1], results.x[0], results.x[numDimensions+1:numDimensions+1+totalCategories]
predicted_diagnol = calculate_probabilities(0, stimulus_representation, w, c, r, categories_idx, b)


plt.style.use('ggplot')
plt.scatter(predicted_dimensional, observed_dimensional, c="r", alpha=0.75, label="Dimensional")
plt.scatter(predicted_crisscross, observed_crisscross, c="b", alpha=0.75, label="Criss-Cross")
plt.scatter(predicted_interiorexterior, observed_interiorexterior, c="g", alpha=0.75, label="Interior-Exterior")
plt.scatter(predicted_diagnol, observed_diagnol, c="c", alpha=0.75, label="Diagnol")
plt.legend(loc='upper left', frameon=True)
plt.xlabel('Augmented GCM Predicted Probabilities')
plt.ylabel('Observed Categorization Probabilities')
plt.title('Subject1')
plt.savefig('../datasets/GCM/nosofsky1986/subject1/AugmentedGCM.png')
plt.show()
