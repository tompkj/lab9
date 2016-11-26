# MIT 6.034 Lab 9: Boosting (Adaboost)
# Written by Jessica Noss (jmn), Dylan Holmes (dxh), and 6.034 staff

from math import log as ln
from utils import *


#### BOOSTING (ADABOOST) #######################################################

def initialize_weights(training_points):
    """Assigns every training point a weight equal to 1/N, where N is the number
    of training points.  Returns a dictionary mapping points to weights."""
    N = len(training_points)
    point_weights = {}
    for p in training_points:
        point_weights[p]= make_fraction(1, N)
    return point_weights

def calculate_error_rates(point_to_weight, classifier_to_misclassified):
    """Given a dictionary mapping training points to their weights, and another
    dictionary mapping classifiers to the training points they misclassify,
    returns a dictionary mapping classifiers to their error rates."""
    classifier_to_error = {}    
    classifiers = classifier_to_misclassified.keys()    
    
    for c in classifiers:  
        misclassified = classifier_to_misclassified[c]
        error_rate = 0
        for m in misclassified:
            error_rate += point_to_weight[m]
        classifier_to_error[c]= error_rate
    return classifier_to_error

def pick_best_classifier(classifier_to_error_rate, use_smallest_error=True):
    """Given a dictionary mapping classifiers to their error rates, returns the
    best* classifier, or raises NoGoodClassifiersError if best* classifier has
    error rate 1/2.  best* means 'smallest error rate' if use_smallest_error
    is True, otherwise 'error rate furthest from 1/2'."""
    c = classifier_to_error_rate
    if use_smallest_error:
        # choose error that is smallest
        smallest_error = min(c, key = c.get)
    else:
        # choose abs error furthest from 1/2
        small_error_below =  min(c, key = c.get)
        small_error_above =  max(c, key = c.get)
        if abs(c[small_error_above]-make_fraction(1,2)) > abs(c[small_error_below]-make_fraction(1,2)):
            smallest_error = small_error_above
        elif abs(c[small_error_above]-make_fraction(1,2)) == abs(c[small_error_below]-make_fraction(1,2)):
            smallest_error = sorted([small_error_above, small_error_below])[0]
        else:
            smallest_error = small_error_below

    if c[smallest_error] == make_fraction(1,2):
        raise NoGoodClassifiersError
    else:
        return smallest_error

def calculate_voting_power(error_rate):
    """Given a classifier's error rate (a number), returns the voting power
    (aka alpha, or coefficient) for that classifier."""
    if error_rate == 1:
        return -INF
    elif error_rate == 0:
        return INF
                
    return make_fraction(0.5*ln((1-error_rate)/error_rate))

def get_overall_misclassifications(H, training_points, classifier_to_misclassified):
    """Given an overall classifier H, a list of all training points, and a
    dictionary mapping classifiers to the training points they misclassify,
    returns a set containing the training points that H misclassifies.
    H is represented as a list of (classifier, voting_power) tuples."""
    misclassified = []
    
    
    for p in training_points:
        vote = 0
        for tup in H:
            classifier = tup[0]
            if p in classifier_to_misclassified[classifier]:
                vote-= tup[1]
            else:
                vote+=tup[1]
            
        if vote <= 0:
            misclassified.append(p)
    
    return set(misclassified)

def is_good_enough(H, training_points, classifier_to_misclassified, mistake_tolerance=0):
    """Given an overall classifier H, a list of all training points, a
    dictionary mapping classifiers to the training points they misclassify, and
    a mistake tolerance (the maximum number of allowed misclassifications),
    returns False if H misclassifies more points than the tolerance allows,
    otherwise True.  H is represented as a list of (classifier, voting_power)
    tuples."""
    set_misclassified = get_overall_misclassifications(H, training_points, classifier_to_misclassified)
    return len(set_misclassified) <= mistake_tolerance

def update_weights(point_to_weight, misclassified_points, error_rate):
    """Given a dictionary mapping training points to their old weights, a list
    of training points misclassified by the current weak classifier, and the
    error rate of the current weak classifier, returns a dictionary mapping
    training points to their new weights.  This function is allowed (but not
    required) to modify the input dictionary point_to_weight."""
    new_weight_dict = {}
    
    for p in point_to_weight.keys():
        old_weight = point_to_weight[p]
        if p not in misclassified_points:
            new_weight = make_fraction(1,2)*1/(1-error_rate)*old_weight
        else:
            new_weight = make_fraction(1,2)*1/(error_rate)*old_weight
        new_weight_dict[p]= new_weight
        
    return new_weight_dict

def adaboost(training_points, classifier_to_misclassified,
             use_smallest_error=True, mistake_tolerance=0, max_rounds=INF):
    """Performs the Adaboost algorithm for up to max_rounds rounds.
    Returns the resulting overall classifier H, represented as a list of
    (classifier, voting_power) tuples."""
    
    point_to_weight = initialize_weights(training_points)
    exit_condition1 = False
    exit_condition2 = False
    exit_condition3 = False
    rounds = 0
    H = []
    
    while not exit_condition1 and not exit_condition2 and not exit_condition3:
        rounds +=1
        classifier_to_error_rate = calculate_error_rates(point_to_weight, classifier_to_misclassified)
        try:
            h = pick_best_classifier(classifier_to_error_rate, use_smallest_error)
        except NoGoodClassifiersError:
            exit_condition3 = True
            
        voting_power = calculate_voting_power(classifier_to_error_rate[h])
        H.append((h, voting_power))
        H_misclassified_points = get_overall_misclassifications(H, training_points, classifier_to_misclassified)
        point_to_weight = update_weights(point_to_weight, H_misclassified_points, classifier_to_error_rate[h])      
        
        exit_condition1 = is_good_enough(H, training_points, classifier_to_misclassified, mistake_tolerance)
        exit_condition2 = (rounds >= max_rounds)
        
    return H
    
#### SURVEY ####################################################################

NAME = "Joseph Lowman"
COLLABORATORS = ""
HOW_MANY_HOURS_THIS_LAB_TOOK = 2
WHAT_I_FOUND_INTERESTING = ""
WHAT_I_FOUND_BORING = ""
SUGGESTIONS = ""
