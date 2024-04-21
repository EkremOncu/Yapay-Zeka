import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
      return 1 / (1 + np.e ** -x)

x = np.linspace(-10, 10, 1000)
y = sigmoid(x)

"""
from tensorflow.keras.activations import sigmoid

y = sigmoid(x).numpy()
"""

plt.title('Sigmoid (Logistic) Function', fontsize=14, fontweight='bold', pad=20)
axis = plt.gca()
axis.spines['left'].set_position('center')
axis.spines['bottom'].set_position('center')
axis.spines['top'].set_color(None)
axis.spines['right'].set_color(None)
axis.set_ylim(-1, 1)
plt.plot(x, y, color='red')
plt.show()

