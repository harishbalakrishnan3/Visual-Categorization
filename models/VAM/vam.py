# Module that contains the necessary functions to implement the VAM
# Author: Harish Balakrishnan


import numpy as np
import copy


def generate_family_of_partitions(list_of_elements):
    """
    Function that generates the family of partitions

    Parameters
    ----------
    list_of_elements : list
        The collection of elements for which we seek the family of partitions. For example,
        if list_of_elements = [1,2,3], then this function will return [[[1, 2, 3]], [[1], [2, 3]], [[1, 2], [3]], [[2],
        [1, 3]], [[1], [2], [3]]]

    Returns
    -------
    list
        List of the family of partitions

    """

    # Thanks to alexis: https://stackoverflow.com/questions/19368375/set-partitions-in-python
    def partition(l):
        if len(l) == 1:
            yield [l]
            return

        first = l[0]
        for smaller in partition(l[1:]):

            # insert `first` in each of the subpartition's subsets
            for n, subset in enumerate(smaller):
                yield smaller[:n] + [[first] + subset] + smaller[n + 1:]
            # put `first` in its own subset
            yield [[first]] + smaller

    family = []
    for p in partition(list_of_elements):
        family.append(p)

    return family


def generate_sub_prototypes(family):
    """
    Function that generates the sub-prototypes or pseudo-exemplars

    Parameters
    ----------
    family : list
        List containing the family of all possible representation models for a given category. For example, if category
        A had five exemplars, then this list should contain all the 52 possible pseudo-exemplar sets.

    Returns
    -------
    list
        List of all sub-prototypes or pseudo-exemplars for the given family of representations of a category.

    """
    # Creating a deep copy of family
    new_family = copy.deepcopy(family)
    bell_no = len(new_family)
    for i in range(bell_no):
        for j in range(len(new_family[i])):
            if len(new_family[i][j]) != 1:
                new_sub = list((np.sum(new_family[i][j], axis=0)) / len(new_family[i][j]))
                new_family[i][j] = [new_sub]

    new_new_family = []

    for i in range(len(new_family)):
        list_of_lists = new_family[i]
        flattened = [val for sublist in list_of_lists for val in sublist]
        new_new_family.append(flattened)

    return new_new_family


def similarity(stimulus_i, stimulus_j, w, c, r, alpha):
    """
    Function that calculates the similarity of stimulus_i and stimulus_j

    Parameters
    ----------
    stimulus_i : list
         Stimulus representation in the psychological space. For example, stimulus_i could be [0, 1, 1, 0]
    stimulus_j : list
        Stimulus representation in the psychological space. For example, stimulus_i could be [1, 0, 0, 1]
    w : list
        List of weights corresponding to each dimension of the stimulus in the psychological space
    c : float
        Scale parameter that is used in the distance calculation
    r : int
        Minkowski’s distance metric. A value of 1 corresponds to city-block metric (generally used when the stimuli has
        separable dimensions) ; A value of 2 corresponds to Eucledian distance metric (generally used when the stimuli
        has integral dimensions)
    alpha : int
        Parameter that scales the psychological distance between stimulus_i and stimulus_j. The value of alpha = 2
        corresponds to the gaussian similarity

    Returns
    -------
    float
        The similarity between stimulus_i and stimulus_j
    """

    def distance():
        """
        Calculates the distance between two stimulus

        Returns
        -------
        np.float64
            Distance scaled by the scale parameter 'c'
        """
        s = 0.0
        N = len(stimulus_i)
        for idx in range(N):
            s += (w[idx] * abs(stimulus_i[idx] - stimulus_j[idx]) ** r)
        s = s ** (1 / r)
        return c * s

    return np.exp(-((distance()) ** alpha))


def similarity_of_i_to_category(stimulus_i, category_exemplars, w, c, r, alpha):
    """
    Function that calculates the similarity of stimulus_i to a particular category exemplars

    Parameters
    ----------
    stimulus_i : list
        Stimulus representation in the psychological space. For example, stimulus_i could be [0, 1, 1, 0]
    category_exemplars : list
        List of category (pseudo)exemplars
    w : list
        List of weights corresponding to each dimension of the stimulus in the psychological space
    c : float
        Scale parameter that is used in the distance calculation
    r : int
        Minkowski’s distance metric. A value of 1 corresponds to city-block metric (generally used when the stimuli has
        separable dimensions) ; A value of 2 corresponds to Eucledian distance metric (generally used when the stimuli
        has integral dimensions)
    alpha : int
        Parameter that scales the psychological distance between stimulus_i and stimulus_j. The value of alpha = 2
        corresponds to the gaussian similarity

    Returns
    -------
    float
        The similarity of stimulus_i to category_exemplars
    """
    N = len(category_exemplars)
    s = 0.0
    for i in range(N):
        s += similarity(stimulus_i, category_exemplars[i], w, c, r, alpha)
    return s


def probability_of_category_A(stimulus_i, category_a_exemplars, category_b_exemplars, w, c, r, alpha, b):
    """
    Function that calculates the probability of stimulus_i belonging to category A. Note that this assumes there are
    only two categories in total, which is typical of categorization experiments

    Parameters
    ----------
    stimulus_i : list
        Stimulus representation in the psychological space. For example, stimulus_i could be [0, 1, 1, 0]
    category_a_exemplars : list
        List of category A (pseudo)exemplars
    category_b_exemplars : list
        List of category B (pseudo)exemplars
    w : list
        List of weights corresponding to each dimension of the stimulus in the psychological space
    c : float
        Scale parameter that is used in the distance calculation
    r : int
        Minkowski’s distance metric. A value of 1 corresponds to city-block metric (generally used when the stimuli has
        separable dimensions) ; A value of 2 corresponds to Eucledian distance metric (generally used when the stimuli
        has integral dimensions)
    alpha : int
        Parameter that scales the psychological distance between stimulus_i and stimulus_j. The value of alpha = 2
        corresponds to the gaussian similarity
    b : list
        List of biases for the categories

    Returns
    -------
    float
        Calculates the probability of stimulus_i belonging to category A
    """
    numerator = b[0] * similarity_of_i_to_category(stimulus_i, category_a_exemplars, w, c, r, alpha)
    denominator = numerator + b[1] * similarity_of_i_to_category(stimulus_i, category_b_exemplars, w, c, r, alpha)
    return numerator / denominator
