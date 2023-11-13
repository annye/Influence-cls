


"""
Desc: This script calculates metrics and performance scores.
      
"""

#import matplotlib.pyplot as plt
import numpy as np
from pdb import set_trace
import pickle
from sklearn.metrics import confusion_matrix, accuracy_score, auc, roc_curve, roc_auc_score


def get_and_record_scores(outer_predictions):
    """Generate a range of relevant performance metrics."""
    results = {}
    all_test = np.concatenate(outer_predictions['Fold test'])
    all_pred = np.concatenate(outer_predictions['Fold predictions'])
    all_probas = np.concatenate(outer_predictions['Fold probabilities'])
    acc = accuracy_score(all_test, all_pred)
    fpr, tpr, thresholds = roc_curve(all_test, all_probas)
    auc_score = auc(fpr, tpr)
    cm = confusion_matrix(all_test, all_pred)
    sens = cm[1][1] / (cm[1][1] + cm[1][0])
    spec = cm[0][0] / (cm[0][0] + cm[0][1])
    prec = cm[1][1] / (cm[1][1] + cm[0][1])
    print('Sens: {}, Spec: {}, Prec: {}, Acc: {}, AUC: {}'.format(sens, spec, prec, acc, auc_score))
    results['auc'] = auc_score; results['sens']  = sens; results['spec'] = spec;
    results['prec'] = prec; results['acc'] = acc
    # plt.plot(fpr, tpr); plt.show()

    results['All test'] = all_test; results['All pred'] = all_pred; results['All probas'] = all_probas
    print('Results: {}'.format(results))
    return results


def save_results_dictionary(outer_results, filepath):
    """Save dictionary of results """
    with open(filepath, 'wb') as f: pickle.dump(outer_results, f)
    

def load_results_dictionary(filepath):
    """Load dictionary of results """
    with open(filepath, 'rb') as f:
        loaded_dict = pickle.load(f)
    return loaded_dict