#!/usr/bin/env python3
"""Function that calculates the F1 score of a confusion matrix"""
import numpy as np


def f1_score(confusion):
    """Calculates the F1 score of a confusion matrix"""
    sensitivity = __import__('1-sensitivity').sensitivity
    precision = __import__('2-precision').precision
    rec = sensitivity(confusion)
    prec = precision(confusion)
    return (2 * rec * prec) / (prec + rec)
