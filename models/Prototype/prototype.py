# Module that contains the necessary functions to implement the Prototype model
# Author: Harish Balakrishnan

import numpy as np

def distance(w, x, i, P, r):
    """
    Function that calculates the distance between stimulus and prototype in the psychological space

    Parameters
    ----------
    w : list
        This is the list of weights corresponding to each dimension of the stimulus in the psychological space
    x : np.array
        This is the stimuli representation in the psychological space. The stimuli are given to this function in the
        form of a n x N matrix, where n is the number of stimuli and N is the number of dimensions of each stimuli in
        the psychological space
    i : int
        Stimulus i
    P : list
        P is the prototype that is represented in the form of co-ordinates in the psychological space. For example,
        if it is a four dimensional psychological space, a prototype for a category could be [1,1,1,1]
    r : int
        This is the Minkowski's distance metric. A value of 1 corresponds to city-block metric (generally used when the
        stimuli has separable dimensions) ; A value of 2 corresponds to eucledian distance metric (generally used when
        the stimuli has integral dimensions)

    Returns
    -------
    float
        The distance between Stimulus i and Prototype P in the psychological space
    """
    total_dimensions = len(w)
    s = 0
    for k in range(total_dimensions):
        s += w[k] * (abs(x[i][k] - P[k])) ** r
    return s ** (1 / r)


def similarity(c, d):
    """
    Function that calculates the similarity between stimulus i and P

    Parameters
    ----------
    c : float
        This is the scale parameter used in the similarity calculation
    d : float
        This is the distance between stimulus and the prototype in the psychological space

    Returns
    -------
    float
        The similarity between the stimulus and the prototype
    """
    return (np.exp(-c * d))


def probability_of_category(categories, category, prototypes, w, x, i, r, c):
    """
    Function that calculates the probability of category response A given stimulus i

    Parameters
    ----------
    categories : list
        This is the list that indicates which stimulus belongs to which category. For example, out of 5 stimuli, if
        stimuli #0, #2, #4 belongs to category 0 and the rest belongs to category 1, then
        categories_Idx = [[0,2,4],[1,3]]
    category : int
        Category number
    prototypes : list
        This is the list that contains the prototype representation for each category. For example, if there are two
        categories and the stimulus representation has four dimensions, the prototypes could be [[1,1,1,1],[0,0,0,0]]
    w : list
        This is the list of weights corresponding to each dimension of the stimulus in the psychological space
    x : np.array
        This is the stimuli representation in the psychological space. The stimuli are given to this function in the
        form of a n x N matrix, where n is the number of stimuli and N is the number of dimensions of each stimuli in
        the psychological space
    i : int
        Stimulus i
    r : int
        This is the Minkowski's distance metric. A value of 1 corresponds to city-block metric (generally used when the
        stimuli has separable dimensions) ; A value of 2 corresponds to eucledian distance metric (generally used when
        the stimuli has integral dimensions)
    c : float
        This is the scale parameter used in the similarity calculation

    Returns
    -------
    float
        The probability of category response A given stimulus i

    """
    # Calculating numerator
    d = distance(w, x, i, prototypes[category], r)
    numerator = similarity(c, d)

    # Calculating denominator
    denominator = 0
    total_categories = len(categories)
    for j in range(total_categories):
        d = distance(w, x, i, prototypes[j], r)
        s = similarity(c, d)
        denominator += s

    if numerator >= denominator:
        print(numerator, denominator)

    return numerator / denominator

