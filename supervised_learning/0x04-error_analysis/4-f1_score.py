#!/usr/bin/env python3
"""Function that calculates the F1 score of a confusion matrix"""
import numpy as np


def f1_score(confusion):
    """Calculates the F1 score of a confusion matrix"""
    tp = np.diagonal(confusion)
    fp = np.sum(confusion, axis=0) - tp
    fn = np.sum(confusion, axis=1) - tp
    return tp / (tp + 0.5 * (fp + fn))
