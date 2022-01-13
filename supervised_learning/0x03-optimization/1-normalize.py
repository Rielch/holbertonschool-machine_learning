#!/usr/bin/env python3
"""Function that normalizes (standardizes) a matrix"""


def normalize(X, m, s):
    """Normalizes (standardizes) a matrix"""
    return (X - m) / s
