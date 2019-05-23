
import os
import csv
import math
import random
import numpy as np
from decimal import Decimal
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist
from matplotlib import style

data = []

path = str(input("What is the path to the folder containing the data : "))
linkage = int(input("What's the linkage method : "))


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

'''
Agglomerative: This is a "bottom up" approach: 
each observation starts in its own cluster, 
and pairs of clusters are merged as one moves up the hierarchy.
'''

# ------------------------------------------------------------------------------
# Hierarchy class
# ------------------------------------------------------------------------------

class hierarchy:
    def __init__(self, data, left_subtree = None, right_subtree = None, distance = 0.0, id = None):
        self.data = data
        self.left_subtree = left_subtree
        self.right_subtree = right_subtree
        self.distance = distance
        self.id = id

# ------------------------------------------------------------------------------
# Hierarchical clustering algorithm
# ------------------------------------------------------------------------------

def hierarchy_clusters(data, distance=cdist):
    distance_cache = []
    current_cluster_id = -1

    print(data.shape)
    # initiate clusters with each row
    # each of the dataset corresponds to one cluster basically
    cluster_list = [hierarchy(data[i], id = i) for i in range(len(data))]


    # we browse all the clusters until there is only one cluster left

    while (len(cluster_list) > 1):
        lowest_pair = (0, 1)
        closest = distance(cluster_list[0].data, cluster_list[1].data)

        for i in range(len(cluster_list)):
            for j in range(i+1, len(cluster_list)):
                if (cluster_list[i].id, cluster[j].id) not in distance_cache:
                    distance_cache[(cluster_list[i].id, cluster_list[j].id)] = distance(cluster_list[i].data, cluster_list[j].data)
                closest_dist = distance_cachedistance_cache[(cluster_list[i].id, cluster_list[j].id)]

                if (closest_dist < closest):
                    closest = closest_dist
                    lowest_pair = (i, j)

        merge_clusters=[
        (cluster[lowest_pair[0]].data_list[i]+cluster[lowest_pair[1]].data_list[i])/2.0
        for i in range(len(cluster[0].data_list))]

        new_cluster = hierarchy(merge_clusters, left_subtree = cluster_list[lowest_pair[0]],
            right_subtree = cluster_list[lowest_pair[1]],
            distance = closest, id=current_cluster_id)

        current_cluster_id-=1
        del cluster_list[lowest_pair[1]]
        del cluster_list[lowest_pair[0]]
        cluster_list.append(new_cluster)

    return cluster_list[0]

hierarchy_clusters(data)