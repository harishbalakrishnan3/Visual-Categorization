# Module that contains the necessary functions to implement the Rational Model of Categorization
# Author: Harish Balakrishnan


def prior_probability_for_new_category(c, nk):
    """
    Function that calculates the prior probability for forming a new category. This is equation 3-5 in [Ande90]_

    Parameters
    ----------
    c : float
        Coupling probability. This is the probability that two objects come from the same category
    nk : list
        List containing the number of objects in each category. For example, if there are two objects in category 0,
        three objects in category 1 and 4 objects in category 2, then nk = [2, 3, 4]

    Returns
    -------
    float
        The prior probability for forming a new category
    """

    n = sum(nk)
    return (1 - c) / (1 - c + c * n)


def prior_probability_of_category_k(k, c, nk):
    """
    Function that calculates the prior probability of category k. This is equation 3-4 in [Ande90]_

    Parameters
    ----------
    k : int
        Category index. For example, k can be 0 or 1 or 2 etc
    c : float
        Coupling probability. This is the probability that two objects come from the same category
    nk : list
        List containing the number of objects in each category. For example, if there are two objects in category 0,
        three objects in category 1 and 4 objects in category 2, then nk = [2, 3, 4]

    Returns
    -------
    float
        The prior probability of category k

    """
    n = sum(nk)
    return (c * nk[k]) / (1 - c + c * n)


def probability_of_ij_given_k(i, j, k, category_membership_list, stimuli, m):
    """
    Function that calculates the mean of the posterior probability density for non-informative priors.
    More specifically, it is the probability that the stimulus displays value j on dimension i given that it comes from
    category k (equation 3-7 in [Ande90]_)

    Parameters
    ----------
    i : int
        Dimension index
    j : int
        Value on dimension i
    k : int
        Category index. For example, k can be 0 or 1 or 2 etc
    category_membership_list : list
        List of lists that contain the category membership details for the stimuli seen so far. For example, if
        stimuli #0, #1, #4 belongs to category #0 and stimuli #2 and #3 belongs to category #1, then
        category_membership_list = [[0, 1, 4], [2, 3]]
    stimuli : list
        List of lists that contain the stimuli seen so far. For example, after training phase of experiment 1 of
        [MeSc78]_, stimuli = [[1, 1, 1, 1, 1], [1, 0, 1, 0, 1], [0, 1, 0, 1, 1], [0, 0, 0, 0, 0], [0, 1, 0, 0, 0],
        [1, 0, 1, 1, 0]]
    m : list
        List that contains the number of unique values in each dimension. For example, if there are five dimensions and
        all the dimensions are binary, then m = [2, 2, 2, 2, 2]

    Returns
    -------
    float
        The probability that the stimulus displays value j on dimension i given that it comes from category k.
    """

    # Denominator
    nk = len(category_membership_list[k])
    denominator = nk + m[i]

    # Numerator
    count = 0
    for idx in range(len(category_membership_list[k])):
        if stimuli[category_membership_list[k][idx]][i] == j:
            count += 1
    numerator = count + 1

    return numerator / denominator


def probability_of_F_given_k(current_stimulus, k, category_membership_list, stimuli, m):
    """
    Function that calculates the probability of observing the current feature set of the stimulus given that it comes
    from category k (equation 3-6 in [Ande90]_)

    Parameters
    ----------
    current_stimulus : list
        Current stimulus representation. For example, current_stimulus could be [1, 0, 0, 1, 0]
    k : int
        Category index. For example, k can be 0 or 1 or 2 etc
    category_membership_list : list
        List of lists that contain the category membership details for the stimuli seen so far. For example, if
        stimuli #0, #1, #4 belongs to category #0 and stimuli #2 and #3 belongs to category #1, then
        category_membership_list = [[0, 1, 4], [2, 3]]
    stimuli : list
        List of lists that contain the stimuli seen so far. For example, after training phase of experiment 1 of
        [MeSc78], stimuli = [[1, 1, 1, 1, 1], [1, 0, 1, 0, 1], [0, 1, 0, 1, 1], [0, 0, 0, 0, 0], [0, 1, 0, 0, 0],
        [1, 0, 1, 1, 0]]
    m : list
        List that contains the number of unique values in each dimension. For example, if there are five dimensions and
        all the dimensions are binary, then m = [2, 2, 2, 2, 2]

    Returns
    -------
    float
        The probability of observing the current feature set of the stimulus given that it comes from category k.
    """

    dimensions = len(current_stimulus)
    product = 1
    for i in range(dimensions):
        product *= probability_of_ij_given_k(i, current_stimulus[i], k, category_membership_list, stimuli, m)

    return product


def probability_of_k_fiven_F(current_stimulus, stimuli, k, category_membership_list, m, c, nk):
    """
    Function that calculates the probability that the current stimulus comes from category k given that it has the
    feature set F (equation 3-3 in [Ande90]_)

    Parameters
    ----------
    current_stimulus : list
        Current stimulus representation. For example, current_stimulus could be [1, 0, 0, 1, 0]
    stimuli : list
        List of lists that contain the stimuli seen so far. For example, after training phase of experiment 1 of
        [MeSc78], stimuli = [[1, 1, 1, 1, 1], [1, 0, 1, 0, 1], [0, 1, 0, 1, 1], [0, 0, 0, 0, 0], [0, 1, 0, 0, 0],
        [1, 0, 1, 1, 0]]
    k : int
        Category index. For example, k can be 0 or 1 or 2 etc
    category_membership_list : list
        List of lists that contain the category membership details for the stimuli seen so far. For example, if
        stimuli #0, #1, #4 belongs to category #0 and stimuli #2 and #3 belongs to category #1, then
        category_membership_list = [[0, 1, 4], [2, 3]]
    m : list
        List that contains the number of unique values in each dimension. For example, if there are five dimensions and
        all the dimensions are binary, then m = [2, 2, 2, 2, 2]
    c : float
        Coupling probability. This is the probability that two objects come from the same category
    nk : list
        List containing the number of objects in each category. For example, if there are two objects in category 0,
        three objects in category 1 and 4 objects in category 2, then nk = [2, 3, 4]

    Returns
    -------
    float
        The probability that the current stimulus comes from category k given that it has the feature set F
    """

    posterior = []
    num_categories = len(category_membership_list)

    for i in range(num_categories):
        posterior.append(prior_probability_of_category_k(i, c, nk) * probability_of_F_given_k(current_stimulus, i,
                                                                                              category_membership_list,
                                                                                              stimuli, m))

    # For the new category
    product = 1
    for i in range(len(m)):
        product *= 1 / m[i]

    posterior.append(prior_probability_for_new_category(c, nk) * product)
    denominator = sum(posterior)

    if k == -1:
        numerator = posterior[-1]
    else:
        numerator = posterior[k]

    return numerator / denominator


def predicted_i_j(i, current_stimulus, category_membership_list, stimuli, m, c, nk):
    """
    Function that predicts value j on dimension i for the current stimulus (equation 3-2 in [Ande90]_)

    Parameters
    ----------
    i : int
        Dimension index
    current_stimulus : list
        Current stimulus representation. For example, current_stimulus could be [1, 0, 0, 1, 0
    category_membership_list : list
        List of lists that contain the category membership details for the stimuli seen so far. For example, if
        stimuli #0, #1, #4 belongs to category #0 and stimuli #2 and #3 belongs to category #1, then
        category_membership_list = [[0, 1, 4], [2, 3]]
    stimuli : list
        List of lists that contain the stimuli seen so far. For example, after training phase of experiment 1 of
        [MeSc78], stimuli = [[1, 1, 1, 1, 1], [1, 0, 1, 0, 1], [0, 1, 0, 1, 1], [0, 0, 0, 0, 0], [0, 1, 0, 0, 0],
        [1, 0, 1, 1, 0]]
    m : list
        List that contains the number of unique values in each dimension. For example, if there are five dimensions and
        all the dimensions are binary, then m = [2, 2, 2, 2, 2]
    c : float
        Coupling probability. This is the probability that two objects come from the same category
    nk : list
        List containing the number of objects in each category. For example, if there are two objects in category 0,
        three objects in category 1 and 4 objects in category 2, then nk = [2, 3, 4]

    Returns
    -------
    float
        Probability of value j on dimension i for the current stimulus
    """

    s = 0
    num_categories = len(category_membership_list)
    for idx in range(num_categories):
        s += probability_of_k_fiven_F(current_stimulus, stimuli, idx, category_membership_list, m, c,
                                      nk) * probability_of_ij_given_k(i, current_stimulus[i], idx,
                                                                      category_membership_list, stimuli, m)

    return s

