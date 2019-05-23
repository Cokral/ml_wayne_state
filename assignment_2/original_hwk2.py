
import csv
import random
import math

# ------------------------------------------------------------------------------
# data initialization
# ------------------------------------------------------------------------------


data = []
with open('creditcard.csv', newline='') as csvfile:
	creditcard = csv.reader(csvfile, delimiter=',', quotechar='|')
	i = 0
	for row in creditcard:
		# we get the first row because it contains the names of the classes and there is no need for it to be left
		if (i == 0):
			first_row = row
			i = 1
		else:
			data.append(row)
	# we have to change the values of the '"0"' and '"1"' so we can use int and float functions later
	for i in range(len(data)):
		for j in range(len(data[i])):
			if(data[i][j] == '"0"'):
				data[i][j] = 0
			elif(data[i][j] == '"1"'):
				data[i][j] = 1
			else:
				data[i][j] = float(data[i][j])

# ------------------------------------------------------------------------------
# training and testing sets
# ------------------------------------------------------------------------------


def splitSetsRandomly(data, ratio):
	size = int(len(data) * ratio)
	training_set = []
	temp = list(data)
	while len(training_set) < size:
		i = random.randrange(len(temp))
		training_set.append(temp.pop(i))
	return [training_set, temp]


# verify that we have the same amount of 0 and 1

# ------------------------------------------------------------------------------
# separate datasets by classes
# ------------------------------------------------------------------------------


# First, we separate the set by class, so that set[0] contains all the variables linked to the 0 class
# and set[1] the others
def separateByClass(dataset):
	separated = {}
	for i in range(len(dataset)):
		vector = dataset[i]
		if (vector[-1] not in separated):
			separated[vector[-1]] = []
		separated[vector[-1]].append(vector)
	return separated


# ------------------------------------------------------------------------------
# calculate mean and standart deviation
# ------------------------------------------------------------------------------


def mean(values):
	return sum(values) / float(len(values))

def standart_deviation(numbers):
	average	 = mean(numbers)
	variance = sum([pow(x - average, 2) for x in numbers]) / float(len(numbers) - 1)
	return math.sqrt(variance)

# ------------------------------------------------------------------------------
# get mean & standart deviation for each variable
# ------------------------------------------------------------------------------


# for each class (0 or 1)
# we calculate both mean and standart deviation for each variable
def summarize(dataset):
	summaries = [(mean(attribute), standart_deviation(attribute)) for attribute in zip(*dataset)]
	del summaries[-1]
	return summaries


# ------------------------------------------------------------------------------
# summarize for each class
# ------------------------------------------------------------------------------


def getClassSummarize(dataset):
	separated = separateByClass(dataset)
	summaries = {}
	for classValue, instances in separated.items():
		summaries[classValue] = summarize(instances)
	return summaries


# ------------------------------------------------------------------------------
# training
# ------------------------------------------------------------------------------


# CONTINOUS

# we use the Gaussian distribution as demonstrated in the lecture
def gaussianDistrib(value, mean, st_dev):
	exp = math.exp(- (math.pow(value - mean, 2) / (2 * math.pow(st_dev, 2))))
	return (1 / (math.sqrt(2 * math.pi) * st_dev)) * exp

# and with the gaussian probability function above, we can calculate the class probabilities
def getClassProbas(summaries, input):
	proba = {}
	for classValue, classSummaries in summaries.items():
		proba[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = input[i]
			proba[classValue] *= gaussianDistrib(x, mean, stdev)
	return proba

# DISCRETE

# P(CK) * P(X1 | CK) * ... * P(Xd | Ck)


# ------------------------------------------------------------------------------
# predictions
# ------------------------------------------------------------------------------


# we know procede to predict the class of a given input
# to do so, we use the given input and the summarize of the data set (mean & standart deviation per class and variable)
# so we first get the probabilities per class (with the function getClassProbas() we made just before)
# and finally we return the class for which the probability is higher
def predict(summaries, input):
	proba 				= getClassProbas(summaries, input)
	classPredicted 		= None
	classProbaPredicted = -1
	for classValue, probability in proba.items():
		if classPredicted == None or probability > classProbaPredicted:
			classPredicted 		= classValue
			classProbaPredicted = probability
	return classPredicted

# now we need to get the predictions for each test of the instance
def getPredictions(summaries, testing_set):
	predictions = []
	for i in range(len(testing_set)):
		predictions.append(predict(summaries, testing_set[i]))
	return predictions


# ------------------------------------------------------------------------------
# calculate precision, recall and F-score
# ------------------------------------------------------------------------------

def getPrecisionRecallFscore(testSet, predictions):
	TP = 0
	FP = 0
	TN = 0
	FN = 0
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			if(predictions[x] == 1):
				TP += 1
			else:
				TN += 1
			correct += 1
		else: 
			if(testSet[x][-1] == 1):
				FP += 1
			else:
				FN += 1
	precision = TP / (FP + TP)
	recall 	  = TN / (TN + FN)
	fscore 	  = 2 * precision * recall / (precision + recall)
	return (precision, recall, fscore)


# ------------------------------------------------------------------------------
# Everything together
# ------------------------------------------------------------------------------

def main():
	# importation of the data is done at the beginning before main
	training_set, testing_set = splitSetsRandomly(data, 0.8)
	# we get the means and standart deviation for each class and each variable
	summaries = getClassSummarize(training_set)
	print(summaries)
	# we get the predictions
	predictions = getPredictions(summaries, testing_set)
	precision, recall, fscore = getPrecisionRecallFscore(testing_set, predictions)
	print("Precision: " + str(precision))
	print("Recall: " + str(recall))
	print("F-score: " + str(fscore))

main()