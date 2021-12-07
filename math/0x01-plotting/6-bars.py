#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
"""plot a stacked bar graph"""


np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))

people = ['Farrah', 'Fred', 'Felicia']

fig, ax = plt.subplots()
width = 0.5

ax.bar(people, fruit[0], width, label='apples', color='red')
ax.bar(people, fruit[1], width, bottom=fruit[0], label='bananas', color='yellow')
ax.bar(people, fruit[2], width, bottom=fruit[1] + fruit[0], label='oranges', color='#ff8000')
ax.bar(people, fruit[3], width, bottom=fruit[2] + fruit[1] + fruit[0], label='peaches', color='#ffe5b4')

ax.set_ylabel('Quantity of Fruit')
plt.ylim([0, 80])
ax.set_title('Number of Fruit per Person')
ax.legend()

plt.show()
