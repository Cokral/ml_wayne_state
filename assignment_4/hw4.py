
import os
import csv
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.metrics.cluster import adjusted_rand_score

style.use('ggplot')
data = []
nbCenters = 21

nbCenters = int(input("How many centers for K-means : "))
path = str(input("What's the file containing the data : "))

if (path == "0"):
    path = "./data/"
# ------------------------------------------------------------------------------
# data initialization
# ------------------------------------------------------------------------------

print('LOADING FILES . . .')
for filename in os.listdir(path):
    print("\tLoading file " + str(filename))
    file = "./processed_data/" + filename
    data_loaded = []
    with open(file, newline='', encoding="utf8") as csvfile:
        dataset = csv.reader(csvfile, delimiter=',', quotechar='|')
        i = 0
        for row in dataset:
            data_loaded.append(row)
    # (np.array(data_loaded).shape)
    data = data + data_loaded

data = np.array(data).astype('float')
data = np.delete(data, 0, 1)
data = np.delete(data, 0, 1)

colors = 10*["g","r","c","b","k"]

print(data.shape)
# print(data)

# plt.scatter(data[:,0], data[:,1], s=15)
# plt.show()

# ------------------------------------------------------------------------------
# K-means class
# ------------------------------------------------------------------------------

class K_Means:
    def __init__(self, k=21, tolerance=0.001, max_iter=1000):
        self.k = k
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.centroids = {}

    def fit(self, data, init):

        # assigne the starting centroids
        print("\tassigning the starting centroids")
        if(len(init) != self.k):
            for i in range(self.k):
                self.centroids[i] = data[i]    
        else:
            for i in range(len(init)):
                self.centroids[i] = data[init[i]]
        
        

        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []
            print("\tcalculating the distances and adding the vectors witht the minimum distance")
            for featureset in data:
                
                distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                print(min(distances))
                # append the cluster closest to the centroid
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)
            # recalculating the clusters centroids
            print("recalculating the clusters centroids")
            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification],axis=0)

            # compare new centroids with old ones to see if we're optimized yet
            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid  = self.centroids[c]
                if np.sum((current_centroid - original_centroid) / original_centroid * 100.0) > self.tolerance:
                    print(np.sum((current_centroid-original_centroid) / original_centroid * 100.0))
                    optimized = False

            if (optimized):
                break;

    def predict(self, data):
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification


# ------------------------------------------------------------------------------
# applying the K-means class to our problem
# ------------------------------------------------------------------------------

print("CREATING KMEANS CLASS WITH " + str(nbCenters) + " CENTERS . . .")
clf = K_Means(nbCenters)
clf.fit(data, [])

for i in range (len(clf.centroids)):
    print("cluster " + str(i))
    print(clf.centroids[i])

data_to_rand = []
for i in range(len(clf.classifications)):
    data_to_rand = data_to_rand + clf.classifications[i]
data_to_rand = np.array(data_to_rand)
data_to_rand = data_to_rand[:,100]



column_true = data[:, 100]


randScore = adjusted_rand_score(column_true, data_to_rand)
print("RAND SCORE: " + str(randScore))


#print("RAND SCORE: 0.012374601037035")