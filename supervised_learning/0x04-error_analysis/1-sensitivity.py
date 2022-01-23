#!/usr/bin/env python3
"""Function that calculates the sensitivity
 for each class in a confusion matrix"""
import numpy as np


def sensitivity(confusion):
    """Caclulates the sesitiviity for
    each class in a confusion matrix"""
    return np.sum(confusion, axis=1) / np.diagonal(confusion)
