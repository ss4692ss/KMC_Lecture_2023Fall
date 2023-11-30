import pandas


import matplotlib.pyplot as pyplot


from sklearn.cluster import KMeans

from sklearn_extra.cluster import KMedoids

from sklearn.metrics import silhouette_score 



dataset = pandas.read_csv("/Users/sanjeevsharma/Desktop/ECON860/dataset.csv")

print(dataset)

dataset = dataset.values

pyplot.scatter(dataset[:,0],dataset[:,1])

pyplot.savefig("scatterplot.png")

pyplot.close()









