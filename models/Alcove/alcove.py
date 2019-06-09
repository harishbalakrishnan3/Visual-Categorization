# Module that contains the necessary functions to implement the ALCOVE model
# Author: Harish Balakrishnan

import numpy as np


def hidden_layer_activations(current_stimulus, stimulus_representation, hidden_representation, alpha, r, q, c):
    """
    Function that calculates the hidden layer activations

    Parameters
    ----------
    current_stimulus : list
        Presenting a stimulus activates all the hidden layer nodes to different extents. The current stimulus is
        represented as co-ordinates in psychological space and is given as a list of length N, where N is the number
        of dimensions of the psychological space. For example, current_stimulus could be [1,0,1,1]
    stimulus_representation : np.array
        The stimuli are given to this function in the form of a n x N matrix, where n is the number of stimuli and N is
        the number of dimensions of each stimuli in the psychological space
    hidden_representation : np.array
        The hidden layer nodes are again represented as co-ordinates in the psychological space. The hidden layer node
        representations are given to this function in the form of a n x N matrix, where n is the number of hidden layer
        nodes and N is the number of dimensions of the psychological space
    alpha : list
        Attentional weights for each dimension. For example, if there are four dimensions in the psychological space and
        if equal attention is assumed for all these dimensions, then alpha is supplied as [0.25, 0.25, 0.25, 0.25]
    r : int
        This is the Minkowski's distance metric. A value of 1 corresponds to city-block metric (generally used when the
        stimuli has separable dimensions) ; A value of 2 corresponds to Eucledian distance metric (generally used when
        the stimuli has integral dimensions)
    q : int
        Similarity Gradient. A value of 1 corresponds to exponential similarity gradient. A value of 2 corresponds to
        Gaussian similarity gradient
    c : float
        Specificity constant. This determines the overall width of the activation profiles of the hidden layer nodes.
        Large values of c results in rapid decrease in similarity and hence narrow activation profiles, whereas small
        values of c results in wide activation profiles. It is one of the free parameters of the model

    Returns
    -------
    list
        A list containing the activations of each node in the hidden layer

    """
    num_hidden_nodes = np.shape(hidden_representation)[0]
    num_dimensions = np.shape(stimulus_representation)[1]
    hidden_activations = []
    for l in range(num_hidden_nodes):
        s = 0
        for k in range(num_dimensions):
            s += alpha[k] * (abs(hidden_representation[l][k] - current_stimulus[k])) ** r
        s = s ** (q / r)
        s *= (-c)
        s = np.exp(s)
        hidden_activations.append(s)
    return hidden_activations


def output_layer_activations(categories, hidden_activations, w):
    """
    Each category is represented as a node in the output layer. This function calculates the activations of each of
    these output category nodes

    Parameters
    ----------
    categories : list
        This is the list that indicates which stimulus belongs to which category. For example, out of 5 stimuli, if
        stimuli #0, #2, #4 belongs to category 0 and the rest belongs to category 1, then
        categories_Idx = [[0,2,4],[1,3]]
    hidden_activations : list
        This is the list that contains all the hidden layer node activations
    w : list
        This is the list of association weights from hidden layer to output layer. If there are J hidden layer nodes and
        K output category nodes, then this list has dimensions J x K. For example, if there are two output category
        nodes and three hidden layer nodes, then w could be [[0.5, 0.2], [0.3, 0.7], [-0.33, -0.77]]

    Returns
    -------
    list
        A list containing the activations of each node in the output category layer

    """
    num_categories = len(categories)
    num_hidden_nodes = len(hidden_activations)
    output_activations = []
    for k in range(num_categories):
        s = 0
        for j in range(num_hidden_nodes):
            s += w[j][k] * hidden_activations[j]
        output_activations.append(s)
    return output_activations


def probability_of_category(K, phi, output_activations):
    """
    Function that calculates the probability of categorizing the current stimulus into category K

    Parameters
    ----------
    K : int
        Category number
    phi : float
        Probability mapping constant. It is a free parameter of the model
    output_activations : list
        This is the list containing the activations of each node in the output category layer

    Returns
    -------
    float
        The probability of categorizing the current stimulus into category K

    """
    num_output_nodes = len(output_activations)
    numerator = np.exp(phi * output_activations[K])
    denominator = 0
    for k in range(num_output_nodes):
        denominator += np.exp(phi * output_activations[k])
    return numerator / denominator


def teacher(i, K, categories, output_activations):
    """
    Feedback in learning is given in the form of teacher values. This is the function that calculates these values

    Parameters
    ----------
    i : int
        Stimulus ID or stimulus number
    K : int
        Category number
    categories : list
        This is the list that indicates which stimulus belongs to which category. For example, out of 5 stimuli, if
        stimuli #0, #2, #4 belongs to category 0 and the rest belongs to category 1, then
        categories_Idx = [[0,2,4],[1,3]]
    output_activations : list
        This is the list containing the activations of each node in the output category layer

    Returns
    -------
    float
        Feedback of the model's performance to the current stimulus in the form of a value that is used in the learning
        phase.
    """
    num_categories = len(categories)
    correct_category = 0
    for k in range(num_categories):
        if i in categories[k]:
            correct_category = k

    if correct_category == K:
        return max(1, output_activations[K])
    else:
        return min(-1, output_activations[K])


def find_del_w(lambda_w, output_activations, hidden_activations, categories):
    """
    Function that calculates the amount of change that should be added to w in the learning phase

    Parameters
    ----------
    lambda_w : float
        Learning rate for the weights. The same learning rate applies to all the weights. It is one of the free
        parameters of the model
    output_activations : list
        This is the list containing the activations of each node in the output category layer
    hidden_activations : list
        This is the list that contains all the hidden layer node activations
    categories : list
        This is the list that indicates which stimulus belongs to which category. For example, out of 5 stimuli, if
        stimuli #0, #2, #4 belongs to category 0 and the rest belongs to category 1, then
        categories_Idx = [[0,2,4],[1,3]]

    Returns
    -------
    float
        The amount of change that should be added to w in the learning phase
    """
    num_categories = len(output_activations)
    num_hidden_layer_nodes = len(hidden_activations)

    del_w = np.zeros([num_hidden_layer_nodes, num_categories], dtype=float)

    for j in range(num_hidden_layer_nodes):
        for k in range(num_categories):
            tr = teacher(j, k, categories, output_activations)
            del_w[j][k] += (lambda_w * (tr - output_activations[k]) * hidden_activations[j])
    return del_w


def find_del_alpha(current_stimulus_id, current_stimulus, lambda_alpha, c, hidden_representation,
                   output_activations, hidden_activations, w, categories):
    """
    Function that calculates the amount of change that should be added to each dimension's attentional weight
    in the learning phase

    Parameters
    ----------
    current_stimulus_id : int
        Current stimulus ID or stimulus number
    current_stimulus : list
        Current stimulus representation in the psychological space. For example, current stimulus could be [0, 1, 1, 0]
    lambda_alpha : float
        Learning rate for the attentional weights. The same learning rate applies to all the weights.
        It is one of the free parameters of the model
    c : float
        Specificity constant. It is one of the free parameters of the model
    hidden_representation : np.array
        The hidden layer nodes are again represented as co-ordinates in the psychological space. The hidden layer node
        representations are given to this function in the form of a n x N matrix, where n is the number of hidden layer
        nodes and N is the number of dimensions of the psychological space
    output_activations : list
        This is the list containing the activations of each node in the output category layer
    hidden_activations : list
        This is the list that contains all the hidden layer node activations
    w : list
        This is the list of association weights from hidden layer to output layer. If there are J hidden layer nodes and
        K output category nodes, then this list has dimensions J x K. For example, if there are two output category
        nodes and three hidden layer nodes, then w could be [[0.5, 0.2], [0.3, 0.7], [-0.33, -0.77]]
    categories : list
        This is the list that indicates which stimulus belongs to which category. For example, out of 5 stimuli, if
        stimuli #0, #2, #4 belongs to category 0 and the rest belongs to category 1, then
        categories_Idx = [[0,2,4],[1,3]]

    Returns
    -------
    list
        A list containing the amount of change that should be added to each dimension's attentional weight in the
        learning phase

    """
    num_categories = len(output_activations)
    num_hidden_layer_nodes = len(hidden_activations)
    num_dimensions = len(current_stimulus)

    del_alpha = []

    for i in range(num_dimensions):
        s2 = 0
        for j in range(num_hidden_layer_nodes):
            s1 = 0
            for k in range(num_categories):
                tr = teacher(current_stimulus_id, k, categories, output_activations)
                s1 += (tr - output_activations[k]) * w[j][k]
            s1 *= hidden_activations[j] * c * abs(hidden_representation[j][i] - current_stimulus[i])
            s2 += s1
        current_alpha = (-lambda_alpha * s2)
        del_alpha.append(current_alpha)
    return del_alpha
