#!/usr/bin/env python3
"""Function that calculates the specificity
 for each class in a confusion matrix"""
import numpy as np


def specificity(confusion):
    """Calculates the specificity for
    each class in a confusion matrix"""
    tp = np.diagonal(confusion)
    fn = np.sum(confusion, axis=1) - tp
    fp = np.sum(confusion, axis=0) - tp
    tn = np.sum(confusion) - (tp + fp + fn)
    return tn / (tn + fp)
