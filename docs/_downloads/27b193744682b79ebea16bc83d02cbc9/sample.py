"""
A sample python script that illustrates how to use the vam module.
As a first step, we need to find the model's parameters - c,w,b (we will assume r = 1 and alpha = 1).
This is done using MLE. After we find the parameters, we display the best category structure for category A and
category B
"""

import pandas as pd, numpy as np
from scipy.optimize import minimize
from vam import *

#######################################################################################################################
# Optimization functions
#######################################################################################################################

def objective(params):
    def SSE(a, b):
        n = len(a)
        sse = 0
        for i in range(n):
            sse += (a[i] - b[i]) ** 2
        return sse

    w, c, b = params[1:numDimensions + 1], params[0], params[numDimensions + 1:numDimensions + 1 + totalCategories]

    probabilities = []

    min_sse = 100

    global min_category_A, min_category_B

    for k in range(15):
        for j in range(52):
            probabilities = []

            # Calculating probabilities for category A
            for i in range(5):
                p = probability_of_category_A(ca[i], prototype_family_a[j], prototype_family_b[k], w, c, 1, 1, b)
                probabilities.append(p)

            # Calculating probabilities for category B
            for i in range(4):
                p = probability_of_category_A(cb[i], prototype_family_a[j], prototype_family_b[k], w, c, 1, 1, b)
                probabilities.append(p)

            sse = SSE(probabilities, training_observed)

            if sse < min_sse:
                min_sse = sse
                min_category_A = prototype_family_a[j]
                min_category_B = prototype_family_b[k]

    return min_sse


def weightConstraint(params):
    weights = params[1:numDimensions + 1]
    sum = 1
    for i in range(numDimensions):
        sum -= weights[i]
    return sum


def biasConstraint(params):
    biases = params[numDimensions + 1:numDimensions + 1 + totalCategories]
    sum = 1
    for i in range(totalCategories):
        sum -= biases[i]
    return sum


def sum_squared_error(a, b):
    n = len(a)
    sse = 0
    for i in range(n):
        sse += (a[i] - b[i]) ** 2
    return sse


#######################################################################################################################
# Pre-processing: Extracting stimulus and generating the family of representations
#######################################################################################################################

# Load the stimuli
data = pd.read_csv('../../datasets/medin1978/stimuli.csv', sep=",")

# Generate the family of representations
ca = list(map(lambda x: list(x), list(data[:5].values)))
family_a = generate_family_of_partitions(ca)

cb = list(map(lambda x: list(x), list(data[5:9].values)))
family_b = generate_family_of_partitions(cb)


# Generate the sub_prototypes family
prototype_family_a = generate_sub_prototypes(family_a)
prototype_family_b = generate_sub_prototypes(family_b)

observed = [0.78, 0.88, 0.81, 0.88, 0.81, 0.16, 0.16, 0.12, 0.03, 0.59, 0.31, 0.94, 0.34, 0.50, 0.62, 0.16]

training_observed = observed[:9]
transfer_observed = observed[9:]


#######################################################################################################################
# Finding the best fit and corresponding model parameters
#######################################################################################################################

# Open parameters
r = 1
alpha = 1

min_category_A = []
min_category_B = []

# Finding the best fit
numDimensions = 4
totalCategories = 2

# Guess Parameters
c = 11.31
w = [0.17, 0.14, 0.24, 0.44]
b = [0.09, 0.91]

# Define Bounds for c, w, b
bounds = []
cBound = (0, np.inf)
bounds.append(cBound)
numWeights = 4
numBiases = 2
for i in range(numWeights):
    bounds.append((0, 1))
for i in range(numBiases):
    bounds.append((0, 1))
bnds = tuple(bounds)

# Define the parameters for the objective function - c, w, b
params = []
params.append(c)
params += w + b

# Constraints
con1 = {'type': 'eq', 'fun': weightConstraint}
con2 = {'type': 'eq', 'fun': biasConstraint}
cons = [con1, con2]

# Minimizing SSE
results = minimize(objective, params, method='SLSQP', bounds=bnds, constraints=cons, options={'disp': True})
w, c, b = results.x[1:numDimensions + 1], results.x[0], results.x[numDimensions + 1:numDimensions + 1 + totalCategories]

# Printing the best fit parameters
print("Best w: ", w)
print("Best c: ", c)
print("Best b: ", b)

# Printing the best fit category representations
print("Minimum Category Representation for A: ", min_category_A)
print("Minimum Category Representation for B: ", min_category_B)



