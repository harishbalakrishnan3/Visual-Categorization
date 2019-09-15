# Module that contains the necessary functions to implement the GCM
# Author: Harish Balakrishnan


import numpy as np


def gaussian_similarity(stimulus_representation, i, j, w, c, r):
    """
    Function that calculates and returns the gaussian similarity of stimuli i and j (equation 4b in [Noso86]_)

    Parameters
    ----------
    stimulus_representation : np.array
        The stimuli are given to this function in the form of a n x N matrix, where n is the number of stimuli and N is
        the number of dimensions of each stimuli in the psychological space
    i : int
        Stimulus i
    j : int
        Stimulus j
    w : list
        This is the list of weights corresponding to each dimension of the stimulus in the psychological space
    c : int
        This is the scale parameter used in the distance calculation
    r : int
        This is the Minkowski's distance metric. A value of 1 corresponds to city-block metric (generally used when the
        stimuli has separable dimensions) ; A value of 2 corresponds to eucledian distance metric (generally used when
        the stimuli has integral dimensions)

    Returns
    -------
    np.float64
        The Gaussian similarity between the two stimulus
    """
    def distance():
        """
        Calculates the distance between two stimulus (equation 6 in [Noso86]_)

        Returns
        -------
        np.float64
            Distance scaled by the scale parameter 'c'
        """
        sum = 0.0
        N = np.shape(stimulus_representation)[1]
        for idx in range(N):
            sum += (w[idx] * (stimulus_representation[i, idx] - stimulus_representation[j, idx]) ** r)
        sum = sum ** (1 / r)
        return c * sum
    return np.exp(-(distance()) ** 2)


def probability_of_category_J(J, stimulus_representation, w, c, r, i, categories_idx, b):
    """
    Function that calculates the probability of category J, given stimulus i (equation 5 in [Noso86]_)

    Parameters
    ----------
    J : int
        Category number
    stimulus_representation : np.array
        The stimuli are given to this function in the form of a n x N matrix, where n is the number of stimuli and N is
        the number of dimensions of each stimuli in the psychological space
    w : list
        This is the list of weights corresponding to each dimension of the stimulus in the psychological space
    c : int
        This is the scale parameter used in the distance calculation
    r : int
        This is the Minkowski's distance metric. A value of 1 corresponds to city-block metric (generally used when the
        stimuli has separable dimensions) ; A value of 2 corresponds to Eucledian distance metric (generally used when
        the stimuli has integral dimensions)
    i : int
        Stimulus i
    categories_idx : list
        This is the list that indicates which stimulus belongs to which category. For example, out of 5 stimuli, if
        stimuli #0, #2, #4 belongs to category 0 and the rest belongs to category 1, then
        categories_Idx = [[0,2,4],[1,3]]
    b : list
        This is the list of biases for the categories

    Returns
    -------
    np.float64
        The probability of stimulus i belonging to category J
    """
    total_categories = len(categories_idx)

    numerator_sum = 0.0
    for temp in range(len(categories_idx[J])):
        numerator_sum += gaussian_similarity(stimulus_representation, i, categories_idx[J][temp], w, c, r)
    numerator_sum *= b[J]

    denominator_sum = 0.0

    for K in range(total_categories):
        denominator_sub_sum = 0
        for temp in range(len(categories_idx[K])):
            denominator_sub_sum += gaussian_similarity(stimulus_representation, i, categories_idx[K][temp], w, c, r)
        denominator_sub_sum *= b[K]
        denominator_sum += denominator_sub_sum

    return numerator_sum / denominator_sum


def calculate_probabilities(J, stimulus_representation, w, c, r, categories_idx, b):
    """
    Function that calculates the probabilities of each of the stimuli belonging to category J. i.e., it calculates
    P(J|i) for all i with a fixed J.
    Note that the function requires all the parameters of the model

    Parameters
    ----------
    J : int
        Category number
    stimulus_representation : np.array
        The stimuli are given to this function in the form of a n x N matrix, where n is the number of stimuli and N is
        the number of dimensions of each stimuli in the psychological space
    w : list
        This is the list of weights corresponding to each dimension of the stimulus in the psychological space
    c : int
        This is the scale parameter used in the distance calculation
    r : int
        This is the Minkowski's distance metric. A value of 1 corresponds to city-block metric (generally used when the
        stimuli has separable dimensions) ; A value of 2 corresponds to Eucledian distance metric (generally used when
        the stimuli has integral dimensions)
    categories_idx : list
        This is the list that indicates which stimulus belongs to which category. For example, out of 5 stimuli, if
        stimuli #0, #2, #4 belongs to category 0 and the rest belongs to category 1, then
        categories_Idx = [[0,2,4],[1,3]]
    b : list
        This is the list of biases for the categories

    Returns
    -------
    list
        The probabilities of all the stimuli being categorised into category J
    """
    probabilities = []
    total_stimuli = np.shape(stimulus_representation)[0]
    for i in range(total_stimuli):
        p = probability_of_category_J(J, stimulus_representation, w, c, r, i, categories_idx, b)
        probabilities.append(p)
    return probabilities

