
import csv
import random
import math
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# data initialization
# ------------------------------------------------------------------------------

data = []
with open('./data/CompleteDataset.csv', newline='', encoding="utf8") as csvfile:
	completeDataset = csv.reader(csvfile, delimiter=',', quotechar='|')
	i = 0
	for row in completeDataset:
		if (i == 0):
			first_row = row
			i = 1
		else:
			data.append(row)

data_numpy = np.array(data)

print("\n--> IMPORTING DATA")
#print("\nFirst row:\n\n" + str(first_row) + "\n\n")

# we need to remove the columns which are not continous

print("--> CLEANING DATA\n")

for i in range(2):
	print("Removing column " + str(first_row[1 - i]) + " . . . ")
	data_numpy = np.delete(data_numpy, 0, 1)
	first_row.pop(1 - i)

for i in range(3):
	print("Removing column " + str(first_row[4 - (i+1)]) + " . . . ")
	data_numpy = np.delete(data_numpy, 1, 1)
	first_row.pop(4 - (i+1))

for j in range(2):
	print("Removing column " + str(first_row[5-i]) + " . . . ")
	data_numpy = np.delete(data_numpy, 3, 1)
	first_row.pop(5 - (i))

print("Removing column " + str(first_row[45]) + " . . . ")
data_numpy = np.delete(data_numpy, 45, 1)
first_row.pop(45)
print("Removing column " + str(first_row[55]) + " . . . ")
data_numpy = np.delete(data_numpy, 55, 1)
first_row.pop(55)

print("Changing values from String to Float . . .")
print("Adding means for the values that are blank . . .")
values = 0
nb_values = 0
for h in range(data_numpy.shape[1]):

	if(values != 0):
		mean = values / nb_values
		for m in range(data_numpy.shape[0]):
			if data_numpy[g, h] == -1:
				data_numpy[g, h] = float(mean)
	values = 0
	nb_values = 0

	for g in range(data_numpy.shape[0]):
		if "+" in data_numpy[g, h]:
			test = data_numpy[g, h].split("+")
			data_numpy[g, h] = float(float(test[0]) + float(test[1]))
			values += float(data_numpy[g, h])
			nb_values += 1
		elif "-" in data_numpy[g, h]:
			test = data_numpy[g, h].split("-")
			data_numpy[g, h] = float(float(test[0]) - float(test[1]))
			values += float(data_numpy[g, h])
			nb_values += 1
		elif h==3 or h==4:
			val = data_numpy[g, h][1:-1]
			if val == '':
				val = 0
			data_numpy[g, h] = val
		elif data_numpy[g, h] == '':
			data_numpy[g, h] = -1
		else:
			data_numpy[g, h] = float(float(data_numpy[g, h]))
			values += float(data_numpy[g, h])
			nb_values += 1

print("Casting all values to Float . . .")
data_numpy = data_numpy.astype(float)
print("\nData cleaned: \n" + str(data_numpy))

# ------------------------------------------------------------------------------
# d-dimensional mean vector
# ------------------------------------------------------------------------------

print("\n---> COMPUTING d-DIMENSIONAL MEAN VECTOR\n")

'''
print("\nnumber of columns: " + str(data_numpy.shape[1]))
print("\nnumber of rows: " + str(data_numpy.shape[0]))
'''

print("Calculating the mean values . . .")

mean_vector = data_numpy.mean(axis=0)
print("\nMean vector : \n")
print(mean_vector)

print("\nCentralize the data . . .\n")
for a in range(data_numpy.shape[1]):
	for z in range(data_numpy.shape[0]):
		data_numpy[z, a] = data_numpy[z, a] - mean_vector[a]
print("\nCentralized data : \n")
print(data_numpy)

# ------------------------------------------------------------------------------
# computing the covariance matrix
# ------------------------------------------------------------------------------

print("\n---> COMPUTING THE COVARIANCE MATRIX\n")

print("Calculating the covariance matrix . . .")

covariance_matrix = np.cov(data_numpy, rowvar=0)
print("\nCovariance matrix :\n")
print(covariance_matrix)
print("\nCovariance matrix shape:\n")
print(covariance_matrix.shape)

# ------------------------------------------------------------------------------
# computing eigenvectors and corresponding eigenvalues
# ------------------------------------------------------------------------------

print("\n---> COMPUTING EIGENVECTORS AND CORRESPONDING EIGENVALUES\n")
# eigenvectors and eigenvalues for the from the covariance matrix
print("Calculating . . .")
eig_val_cov, eig_vec_cov = np.linalg.eig(covariance_matrix)

print("\nEigenvector : \n" + str(eig_vec_cov))
print("\nEigenvalues : \n" + str(eig_val_cov))

# ------------------------------------------------------------------------------
# sorting the eigenvectors by decreasing eigenvalues
# ------------------------------------------------------------------------------

print("\n---> SORTING EIGENVECTORS BY DECREASING EIGENVALUES\n")

print("Making list of (eigenvalue, eigenvector) tuples . . . \n")
eig_pairs = [(np.abs(eig_val_cov[i]), eig_vec_cov[:,i]) for i in range(len(eig_val_cov))]

print("Sorting the tuples . . .\n")
eig_pairs.sort(key=lambda x: x[0], reverse=True)

print("Eigenvalues sorted: \n")
sumEig = 0
for i in eig_pairs:
	print(i[0])
	sumEig += i[0]

for i in eig_pairs[1]:
	print("%variables : " + str(i))

# ------------------------------------------------------------------------------
# choosing k eigenvectors with the largest eigenvalues
# ------------------------------------------------------------------------------

print("\n---> CHOOSING K EIGENVECTORS WITH LARGEST EIGENVALUES\n")
print("Check how much information is on every eigenvalues . . .\n")
print("Eigenvalues by percentage : \n")
l = 0

for o in eig_pairs:
	print(str(l) + " : %.2f" % (o[0] * 100 / sumEig) + " %")
	l += 1

print("Taking the two best eigenvalues . . .\n")
matrix_w = np.hstack((eig_pairs[0][1].reshape(66,1), eig_pairs[1][1].reshape(66,1)))
#matrix_w = (eig_pairs[0][1], eig_pairs[1][1])
print('Matrix W : \n' + str(matrix_w))

list2 = first_row
list1 = eig_pairs[0][1]
list1, list2 = zip(*sorted(zip(abs(list1), list2),reverse=True))

print("\nVariable weight for eigenvector 1 : \n")
for i in range(len(list1)):
	print(str(list2[i]) + " : " + str(list1[i]))

list2 = first_row
list1 = eig_pairs[1][1]
list1, list2 = zip(*sorted(zip(abs(list1), list2),reverse=True))

print("\nVariable weight for eigenvector 2 : \n")
for i in range(len(list1)):
	print(str(list2[i]) + " : " + str(list1[i]))

# ------------------------------------------------------------------------------
# Transforming the samples onto the new subspace
# ------------------------------------------------------------------------------

print("\n--->TRANSFORMING THE SAMPLES ONTO THE NEW SUBSPACE\n")

print("Checking the spaces of the different matrix . . .\n")
print("Shape matrix_x : " + str(matrix_w.shape))
print("Shape matrix_x transposed : " + str(matrix_w.T.shape))
print("Shape data : " + str(data_numpy.shape))

print("Transorming the samples . . . \n")
transformed = matrix_w.T.dot(data_numpy.T)
print("Shape obtained : \n")
print(transformed.shape)
print("\nMatrix obtained : \n")
print(transformed)

# ------------------------------------------------------------------------------
# Plotting the results
# ------------------------------------------------------------------------------

print(max(transformed[0]))
print(min(transformed[0]))
print(max(transformed[1]))
print(min(transformed[1]))

plt.plot(transformed[0,0:17981], transformed[1,0:17981], 'o', markersize=7, color='blue', alpha=0.5, label='players')
plt.xlim([-1200,850])
plt.ylim([-800,850])
plt.xlabel('principal component 1')
plt.ylabel('principal component 2')
plt.legend()
plt.title('Transformed data')

plt.show()