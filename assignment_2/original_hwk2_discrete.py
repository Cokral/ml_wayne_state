
import csv
import random
import math

# ---------------------------------------------------------------------
# data initialization
# ---------------------------------------------------------------------

with open('./data/creditcard.csv', newline='') as csvfile:
  credit_card = csv.reader(csvfile, delimiter=',', quotechar='|')
  i = 0
  first_row = next(credit_card)
  data = [row for row in credit_card]

  for i in range(len(data)):
    for j in range(len(data[i])):

      if(data[i][j] == '"0"'):
        data[i][j] = 0
      elif(data[i][j] == '"1"'):
        data[i][j] = 1
      else:
        value = float(data[i][j])
        if(j == 0):
          if(value < 86396):
            value = "near"
          else:
            value = "far"
        elif(j == 29):
          if(value <= 9.9):
            value = "low"
          elif(9.9 < value <= 50):
            value = "medium"
          else:
            value = "high"
        data[i][j] = value
print(first_row)

# ---------------------------------------------------------------------
# training and testing sets
# ---------------------------------------------------------------------


def splitSetsRandomly(data, ratio):
  size = int(len(data) * ratio)
  training_set = []
  temp = list(data)
  while len(training_set) < size:
    i = random.randrange(len(temp))
    training_set.append(temp.pop(i))
  return [training_set, temp]

# training_set, testing_set = splitSetsRandomly(data, 0.8)


# ---------------------------------------------------------------------
# calculate mean and standart deviation
# ---------------------------------------------------------------------


def mean(values):
  return sum(values) / float(len(values))

def standart_deviation(numbers):
  average  = mean(numbers)
  variance = sum([pow(x - average, 2) for x in numbers]) / float(len(numbers) - 1)
  return math.sqrt(variance)

# ---------------------------------------------------------------------
# separate datasets by classes
# ---------------------------------------------------------------------


# First, we separate the set by class, so that set[0] contains all the 
# variables linked to the 0 class
# and set[1] the others
def separateByClass(dataset):
  separated = {}
  for i in range(len(dataset)):
    vector = dataset[i]
    if (vector[-1] not in separated):
      separated[vector[-1]] = []
    separated[vector[-1]].append(vector)
  return separated

# separated = separateByClass(training_set)
# print(separated)

# ---------------------------------------------------------------------
# calculate mean and standart deviation
# ---------------------------------------------------------------------


def mean(values):
  return sum(values) / float(len(values))

def standart_deviation(numbers):
  average  = mean(numbers)
  variance = sum([pow(x - average, 2) for x in numbers]) / float(len(numbers) - 1)
  return math.sqrt(variance)

# 0 : time
# 1 : amount
def discrete(values, which):
  #for i in range(len(values)):
  # print(values[i])
  means=[]
  if(which == 0):
    means = [['near',0],['far',0]]
    near  = 1
    far   = 1
    for i in range(len(values)):
      if values[i] == "near":
        near += 1
      elif values[i] == "far":
        far += 1
    near = near / len(values)
    far = far / len(values)
    means[0][1] = near
    means[1][1] = far

  else:
    means = [['low', 0], ['medium', 0], ['high', 0]]
    low    = 1
    medium = 1
    high   = 1
    for i in range(len(values)):
      if values[i] == "low":
        low += 1
      elif values[i] == "medium":
        medium += 1
      elif values[i] == "high":
        high += 1
    low = low / len(values)
    medium = medium / len(values)
    high = high / len(values)
    means[0][1] = low
    means[1][1] = medium
    means[2][1] = high
  return means

  # we need something like


# ---------------------------------------------------------------------
# get mean & standart deviation for each variable
# ---------------------------------------------------------------------


# for each class (0 or 1)
# we calculate both mean and standart deviation for each variable
def summarize(dataset):
  summaries = [None] * len(dataset[0])
  which = 0
  for attribute in zip(*dataset):
    if which == 0:
      summaries[which] = (discrete(attribute, 0))
    elif which == 29:
      summaries[which] = (discrete(attribute, 1))
    else:
      summaries[which] = ([mean(attribute), 
                standart_deviation(attribute)])
    which += 1
  del summaries[-1]
  return summaries


# TODO change in case discrete instead of continous

# ------------------------------------------------------------------------------
# summarize for each class
# ------------------------------------------------------------------------------


def getClassSummarize(dataset):
  separated = separateByClass(dataset)
  summaries = {}
  for classValue, instances in separated.items():
    summaries[classValue] = summarize(instances)
  return summaries


# ---------------------------------------------------------------------
# training
# ---------------------------------------------------------------------


# CONTINOUS

# we use the Gaussian distribution as demonstrated in the lecture
def gaussianDistrib(value, mean, st_dev):
  exp = math.exp(- (math.pow(value - mean, 2) / (2 * math.pow(st_dev, 2))))
  return (1 / (math.sqrt(2 * math.pi) * st_dev)) * exp

# and with the gaussian probability function above, we can 
# calculate the class probabilities
def getClassProbas(summaries, input):
  proba = {}
  for classValue, classSummaries in summaries.items():
    proba[classValue] = 1
    for i in range(len(classSummaries)):
      x = input[i]
      if(i == 0):
        probaNear = classSummaries[i][0]
        probaFar = classSummaries[i][1]
        if input == 'near':
          proba[classValue] *= probaNear
        elif input == 'far':
          proba[classValue] *= probaFar
      elif(i == 29):
        probaLow = classSummaries[i][0]
        probaMedium = classSummaries[i][1]
        probaHigh = classSummaries[i][2]
        if input == 'low':
          proba[classValue] *= probaLow
        elif input == 'medium':
          proba[classValue] *= probaMedium
        elif input == 'high':
          proba[classValue] *= probaHigh
      else:
        mean, stdev = classSummaries[i]
        proba[classValue] *= gaussianDistrib(x, mean, stdev)
  return proba



# ---------------------------------------------------------------------
# predictions
# ---------------------------------------------------------------------


# we know procede to predict the class of a given input
# to do so, we use the given input and the summarize of the data set 
# (mean & standart deviation per class and variable)
# so we first get the probabilities per class (with the function 
# getClassProbas() we made just before)
# and finally we return the class for which the probability is higher
def predict(summaries, input):
  proba         = getClassProbas(summaries, input)
  classPredicted    = None
  classProbaPredicted = -1
  for classValue, probability in proba.items():
    if classPredicted == None or probability > classProbaPredicted:
      classPredicted    = classValue
      classProbaPredicted = probability
  return classPredicted

# now we need to get the predictions for each test of the instance
def getPredictions(summaries, testing_set):
  predictions = []
  for i in range(len(testing_set)):
    predictions.append(predict(summaries, testing_set[i]))
  return predictions


# ---------------------------------------------------------------------
# calculate precision, recall and F-score
# ---------------------------------------------------------------------

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
  recall    = TN / (TN + FN)
  fscore    = 2 / ((1 / recall) + (1 / precision))
  return (precision, recall, fscore)


# ---------------------------------------------------------------------
# Everything together
# ---------------------------------------------------------------------

def main():
  # importation of the data is done at the beginning before main
  training_set, testing_set = splitSetsRandomly(data, 0.8)
  # we get the means and standart deviation for each class and each 
  # variable
  summaries = getClassSummarize(training_set)
  print(summaries)
  # we get the predictions
  predictions = getPredictions(summaries, testing_set)
  precision, recall, fscore = getPrecisionRecallFscore(testing_set,
   predictions)
  print("Precision: " + str(precision))
  print("Recall: " + str(recall))
  print("F-score: " + str(fscore))

main()