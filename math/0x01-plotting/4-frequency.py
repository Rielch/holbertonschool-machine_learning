#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
"""plot a histogram of student scores for a project"""


np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

plt.xlim([0, 100])
plt.ylim([0, 30])
plt.title('Project A')
plt.xlabel('Grades')
plt.ylabel('Number of Students')
plt.hist(student_grades, bins=np.arange(0, 110, 10), edgecolor='k')
plt.show()
