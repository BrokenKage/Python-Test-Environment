import NeuralNetwork as nn
import numpy as np
import os
import matplotlib.pyplot as plt

path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'mnist.npz')
with np.load(path) as data:
    training_images = data['training_images']
    training_labels = data['training_labels']

#plt.imshow(training_images[0].reshape(28,28), cmap = 'gray')
#plt.show()

layer_sizes = (784,5,10)

net = nn.NeuralNetwork(layer_sizes)
net.print_accuracy(training_images, training_labels)
