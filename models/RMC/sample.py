"""
A sample python script that illustrates how to use the rmc module. First the model is trained using the training set
stimuli. The category structure is built gradually during training by assigning the stimulus to the category that yields
the maximum joint probability P(k, F). In the test phase, the stimuli are fed into the
model one by one we predict its last dimension, which is the category membership.
"""


from rmc import *
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


#######################################################################################################################
# Step 1: Training Phase. Generating the category structure
#######################################################################################################################

nk = []
category_membership_list = []
stimuli = []
m = [2, 2, 2, 2, 2]
c = 0.5


training_set = [[1, 1, 1, 1, 1], [1, 0, 1, 0, 1], [0, 1, 0, 1, 1], [0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [1, 0, 1, 1, 0]]

for idx in range(len(training_set)):
    current_stimulus = training_set[idx]
    stimuli.append(current_stimulus)
    old_num_categories = len(category_membership_list)

    # Generate the joint posterior list for P(k,F)
    posterior_probabilities = []

    for i in range(len(category_membership_list)):
        p = probability_of_k_fiven_F(current_stimulus, stimuli, i, category_membership_list, m, c, nk)
        posterior_probabilities.append(p)
    else:
        posterior_probabilities.append(
            probability_of_k_fiven_F(current_stimulus, stimuli, -1, category_membership_list, m, c, nk))

    # Assigning the current stimulus to the category that yields the maximum of the joint probability P(k, F)
    cat_of_max_joint_probability = posterior_probabilities.index(max(posterior_probabilities))

    if (cat_of_max_joint_probability == old_num_categories):
        nk.append(1)
        category_membership_list.append([idx])
    else:
        nk[cat_of_max_joint_probability] += 1
        category_membership_list[cat_of_max_joint_probability].append(idx)

# Printing the category structure
for i in range(len(category_membership_list)):
    temp_str = ""
    for j in range(len(category_membership_list[i])):
        temp_str += str(training_set[category_membership_list[i][j]]) + " "
    print("Category " + str(i+1) + ": " + temp_str)

#######################################################################################################################
# Test phase
#######################################################################################################################
test_set = [[1, 1, 1, 1, 1], [0, 1, 0, 1, 1], [1, 0, 1, 0, 1], [1, 1, 0, 1, 1], [0, 1, 1, 1, 1], [0, 0, 0, 1, 1],
            [1, 1, 1, 0, 1], [1, 0, 0, 0, 1], [0, 0, 1, 0, 1], [1, 0, 1, 1, 1], [0, 1, 0, 0, 1], [0, 0, 0, 0, 1]]

predicted_1 = []
c = 0.45
for i in range(12):
    predicted_1.append(predicted_i_j(4, test_set[i], category_membership_list, training_set, m, c, nk))

# Plotting
figure(num=None, figsize=(9, 7), dpi=300, facecolor='w', edgecolor='k')
plt.style.use('ggplot')
x = ["1111_", "0101_", "1010_", "1101_", "0111_", "0001_", "1110_", "1000_", "0010_", "1011_", "0100_", "0000_"]

plt.plot(x, predicted_1, '-p', alpha=0.75, label="c=0.45")
plt.legend(loc='upper left', frameon=True)
plt.xlabel('Stimuli')
plt.ylabel('Estimated Probability of 1')
plt.title("Estimated probability of category 1 in experiment 1 of Medin & Schaffer (1978)")
plt.savefig('../../datasets/medin1978/figure2.png')
