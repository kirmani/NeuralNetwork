# Back-Propagation Neural Networks
# 
# Written in Python.  See http://www.python.org/
# Placed in the public domain.
# Neil Schemenauer <nas@arctrix.com>

import math
import random
import string

import urllib2
from bs4 import BeautifulSoup

random.seed(0)

# calculate a random number where:  a <= rand < b
def rand(a, b):
    return (b-a)*random.random() + a

# Make a matrix (we could use NumPy to speed this up)
def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill]*J)
    return m

# our sigmoid function, tanh is a little nicer than the standard 1/(1+e^-x)
def sigmoid(x):
    return math.tanh(x)

# derivative of our sigmoid function, in terms of the output (i.e. y)
def dsigmoid(y):
    return 1.0 - y**2

class NN:
    def __init__(self, ni, nh, no):
        # number of input, hidden, and output nodes
        # nl is number of hidden layers
        self.ni = ni + 1 # +1 for bias node
        self.nh = nh
        self.no = no
        self.nl = len(nh) + 2

        self.activation = []
        self.activation.append([1.0]*self.ni)
        for i in range(self.nl - 2):
            self.activation.append([1.0]*self.nh[i])
        self.activation.append([1.0]*self.no)

        self.weights = []
        for i in range(1, self.nl):
            # print(len(self.activation[i-1]))
            self.weights.append(makeMatrix(len(self.activation[i-1]), len(self.activation[i])))

        for weight in self.weights:
            for row in range(len(weight)):
                for col in range(len(weight[0])):
                    weight[row][col] = rand(-0.2, 0.2)

        self.change = []
        for i in range(1, self.nl):
            self.change.append(makeMatrix(len(self.activation[i-1]), len(self.activation[i])))

    def update(self , inputs):
        if len(inputs) != self.ni-1:
            raise ValueError('wrong number of inputs')

        # input activation
        for i in range(self.ni - 1):
            self.activation[0][i] = inputs[i]

        for l in range(1, self.nl):
            for j in range(len(self.activation[l])):
                sum = 0.0
                for i in range(len(self.activation[l-1])):
                    sum = sum + self.activation[l-1][i] * self.weights[l-1][i][j]
                self.activation[l][j] = sigmoid(sum)

        return self.activation[len(self.activation) - 1][:]
        
    def backPropagate(self, targets, N, M):
        if len(targets) != self.no:
            raise ValueError('wrong number of target values')

        deltas = []
        deltas.append([0.0]*self.ni)
        for i in range(self.nl-2):
            deltas.append([0.0]*self.nh[i])
        deltas.append([0.0]*self.no)

        for k in range(self.no):
            error = targets[k]-self.activation[len(self.activation) - 1][k]
            deltas[len(self.activation) - 1][k] = dsigmoid(self.activation[len(self.activation) - 1][k]) * error

        l = self.nl - 1
        while l >= 1:
            for tail in range(len(self.activation[l-1])):
                error = 0.0
                for head in range(len(self.activation[l])):
                    error = error + deltas[l][head] * self.weights[l-1][tail][head]
                deltas[l - 1][tail] = dsigmoid(self.activation[l - 1][tail]) * error
                # print(deltas[l - 1][tail])
            l -= 1

        l = self.nl - 1
        while l >= 1:
            for tail in range(len(self.activation[l-1])):
                for head in range(len(self.activation[l])):
                    change = deltas[l][head] * self.activation[l-1][tail]
                    self.weights[l - 1][tail][head] = self.weights[l - 1][tail][head] + N * change + M * self.change[l - 1][tail][head]
                    self.change[l - 1][tail][head] = change
            l -= 1

       # calculate error
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5*(targets[k]-self.activation[len(self.activation) - 1][k])**2
        return error


    def test(self, patterns):
        for p in patterns:
            print(p[0], '->', self.update(p[0]), '->', p[1])

    def weights(self):
        print('Input weights:')
        for i in range(self.ni):
            print(self.wi[i])
        print()
        print('Output weights:')
        for j in range(self.nh):
            print(self.wo[j])

    def getWeights(self):
        return self.weights

    def train(self, patterns, iterations=1000, N=0.4, M=0.1):
        # N: learning rate
        # M: momentum factor
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backPropagate(targets, N, M)
            if i % 100 == 0:
                print('error %-.5f' % error)

def demo():
    # Teach network XOR function
    pat = [
        [[0,0], [0]],
        [[0,1], [1]],
        [[1,0], [1]],
        [[1,1], [0]]
    ]


    # create a network with two input, two hidden layers of two nodes, and one output nodes
    n = NN(2, [2, 2], 1)
    # train it with some patterns
    n.train(pat, 1000)
    # # test it
    n.test(pat)

if __name__ == '__main__':
    demo()