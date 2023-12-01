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

def run_kmeans(n, dataset):
  machine = KMeans(n_clusters=n)
  machine.fit(dataset)
  results = machine.predict(dataset)
  centroids = machine.cluster_centers_
  pyplot.scatter(dataset[:,0],dataset[:,1], c=results)
  pyplot.scatter(centroids[:,0], centroids[:, 1], c="red", marker="*", s=300)
  pyplot.savefig("scatterplot_kmeans_" + str(n) + ".png")
  pyplot.close()
  return silhouette_score(dataset, results, metric="euclidean")


n_list = [2,3,4,5,6,7,8]
silhouette_score_list = [run_kmeans(i, dataset) for i in n_list]

pyplot.scatter(n_list, silhouette_score_list)
pyplot.savefig("silhouette_score_kmeans.png")
pyplot.close()











