import numpy as np

def k_means(X, k, max_iters=100):

  centroids = X[np.random.choice(X.shape[0], k, replace=False)]

  for _ in range(max_iters):
    labels = assign_clusters(X, centroids)
    new_centroids = update_centroids(X, labels, k)

    if np.array_equal(centroids, new_centroids):
      break

    centroids = new_centroids

  return centroids, labels

def assign_clusters(X, centroids):

  labels = np.zeros(X.shape[0], dtype=int)
  for i, x in enumerate(X):
    distances = np.linalg.norm(x - centroids, axis=1)
    labels[i] = np.argmin(distances)

  return labels

def update_centroids(X, labels, k):

  new_centroids = np.zeros((k, X.shape[1]))
  for i in range(k):
    cluster_points = X[labels == i]
    new_centroids[i] = np.mean(cluster_points, axis=0)

  return new_centroids

with open("100k5.arff", "r") as f:
  lines = f.readlines()
  data_start = lines.index("@data\n") + 1
  data = []
  for line in lines[data_start:]:
    data.append([float(x) for x in line.strip().split(",")])
X = np.array(data)


N = X.shape[0]  
D = X.shape[1]  
k = 6 

centroids, labels = k_means(X, k)

for i in range(D):
  print(f"X{i+1}: {centroids[:, i]}") 

print("Cluster labels:", labels)
