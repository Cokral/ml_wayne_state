
from numpy import *

# we use a class because it's easier to save the weights, rules and everything
class AdaBoost:
	# when we intantiate the object, we need the training data set already
    def __init__(self, training_set):
        self.training_set = training_set
        self.N = len(self.training_set)
        self.weights = ones(self.N)/self.N
        self.RULES = []
        self.ALPHA = []

    # add_hypothesis is the main method that follows the whole adaboost algorithm
    def add_hypothesis(self, func, test=False):
    	# first we calculate the errors
        errors = array([t[1]!=func(t[0]) for t in self.training_set])
        e = (errors*self.weights).sum()
        # and with that error we can compute the alpha 
        alpha = 0.5 * log((1-e)/e)
        print("error of the hypothesis: " + str(e))
        print("alpha (weight of the hypothesis): " + str(alpha))

        # we will now determine the new weights
        # we first instantiate the empty array
        w = zeros(self.N)
        for i in range(self.N):
            if errors[i] == 1: w[i] = self.weights[i] * exp(alpha)
            else: w[i] = self.weights[i] * exp(-alpha)
        # we make sure to normalize the weights
        self.weights = w / w.sum()
        self.RULES.append(func)
        self.ALPHA.append(alpha)

    def evaluate(self):
        NR = len(self.RULES)
        for (x,l) in self.training_set:
            hx = [self.ALPHA[i]*self.RULES[i](x) for i in range(NR)]
            if(sign(l) == sign(sum(hx))): print(x)

# creating the dataset
data = []
data.append(((11,3), -1))
data.append(((10,1), -1))
data.append(((4,4), -1))
data.append(((12,10), +1))
data.append(((2,4), -1))
data.append(((10,5), +1))
data.append(((8,8), -1))
data.append(((6,5), +1))
data.append(((7,7), +1))
data.append(((7,8), +1))

# function used to print the weights 
print(data)
def print_tab(tab):
	for i in range(len(tab)):
		print(str(i+1) + " | " + str(tab[i]))

# creating the object adaboost with the dataset created
m = AdaBoost(data)
# we had new rules, and print the weights each time to see the evolution
print_tab(m.weights)
m.add_hypothesis(lambda x: 2*(x[1] > 4)-1)
print_tab(m.weights)
m.add_hypothesis(lambda x: 2*(x[0]%2 != 0)-1)
print_tab(m.weights)
m.add_hypothesis(lambda x: 2*(x[0] > x[1] and x[1] >= 5)-1)
print_tab(m.weights)

# finally we can evaluate again
m.evaluate()