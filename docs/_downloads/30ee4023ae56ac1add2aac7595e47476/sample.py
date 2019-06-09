"""
A sample python script that illustrates how to use the alcove module.
The dataset by Shepard et al. 1961 is used. Six different category structures are explored. The plot in figure 5 of
Kruschke 1992 is generated in the end
"""

import pandas as pd
import matplotlib.pyplot as plt
from alcove import *

def generate_Pr(categories):
    """
    Function that returns the Probability(Correct) given category memberships

    Parameters
    ----------
    categories : list
        This is the list that indicates which stimulus belongs to which category. For example, out of 5 stimuli, if
        stimuli #0, #2, #4 belongs to category 0 and the rest belongs to category 1, then
        categories_Idx = [[0,2,4],[1,3]]

    Returns
    -------
    list
        A list of fifty Pr(Correct) values as the model learned association weights and attentional weights
    """
    # Initializing free parameters
    c = 6.5
    phi = 2.0
    lambda_w = 0.03
    lambda_alpha = 0.0033
    # lambda_alpha = 0

    # Initializing Hyper parameters
    r = 1
    q = 1
    alpha = [0.3333, 0.3333, 0.3333]
    w = [[0.0, 0.0] for i in range(8)]

    Pr = []

    Pr_sum = 0
    for epochs in range(400):

        current_iteration = epochs
        if current_iteration >= 8:
            current_iteration = current_iteration % 8
        current_stimulus = stimuli[current_iteration]

        hidden_activations = hidden_layer_activations(current_stimulus, stimuli, stimuli, alpha, r, q, c)
        output_activations = output_layer_activations(categories, hidden_activations, w)

        correct = 0
        # Finding what is correct
        for k in range(len(output_activations)):
            if current_iteration in categories[k]:
                correct = k


        p = probability_of_category(correct, phi, output_activations)
        Pr_sum += p

        if (current_iteration + 1) % 8 == 0:
            Pr_sum /= 8
            Pr.append(Pr_sum)
            Pr_sum = 0

        # Online Learning of w and alpha
        del_w = find_del_w(lambda_w, output_activations, hidden_activations, categories)
        for j in range(len(hidden_activations)):
            for k in range(len(output_activations)):
                w[j][k] += del_w[j][k]
        del_alpha = find_del_alpha(current_iteration, current_stimulus,  lambda_alpha, c, stimuli,
                                   output_activations, hidden_activations, w, categories)

        for i in range(3):
            alpha[i] += del_alpha[i]
            if (alpha[i] < 0):
                alpha[i] = 0

    return Pr


# Stimulus
data = pd.read_csv('../../datasets/shepard1961/stimuli.csv', sep=",")
stimuli = data.values

# Type 1 Categorization
categories1 = [[0, 1, 2, 3], [4, 5, 6, 7]]
Pr_type1 = generate_Pr(categories1)

# Type 2 Categorization
categories2 = [[0, 1, 6, 7], [2, 3, 4, 5]]
Pr_type2 = generate_Pr(categories2)

# Type 3 Categorization
categories3 = [[0, 1, 2, 5], [3, 4, 6, 7]]
Pr_type3 = generate_Pr(categories3)

# Type 4 Categorization
categories4 = [[0, 1, 2, 4], [3, 5, 6, 7]]
Pr_type4 = generate_Pr(categories4)

# Type 5 Categorization
categories5 = [[0, 1, 2, 7], [3, 4, 5, 6]]
Pr_type5 = generate_Pr(categories5)

# Type 6 Categorization
categories6 = [[0, 3, 5, 6], [1, 2, 4, 7]]
Pr_type6 = generate_Pr(categories6)

x = [i+1 for i in range(len(Pr_type6))]
plt.style.use('bmh')
plt.plot(x, Pr_type1, alpha=0.75, label="Type-I")
plt.plot(x, Pr_type2, alpha=0.75, label="Type-II")
plt.plot(x, Pr_type3, alpha=0.75, label="Type-III")
plt.plot(x, Pr_type4, alpha=0.75, label="Type-IV")
plt.plot(x, Pr_type5, alpha=0.75, label="Type-V")
plt.plot(x, Pr_type6, alpha=0.75, label="Type-VI")
plt.legend(loc='center right', frameon=True, title="Category Structure")
plt.xlabel('Epoch')
plt.ylabel('Pr(correct)')
plt.savefig('../../datasets/shepard1961/sample.png')
plt.show()
