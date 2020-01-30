import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

cluster1=np.random.randn(300,2,)+np.array([1,1])
cluster2=np.random.randn(300,2,)+np.array([4,7])
cluster3=np.random.randn(300,2,)+np.array([3,3])
cluster4=np.random.randn(300,2,)+np.array([7,4])

data=np.r_[cluster1,cluster2,cluster3,cluster4]

x_train, x_test=train_test_split(data, test_size=0.33)

kmeans=KMeans(n_clusters=4)
kmeans.fit(x_train)
clusters=kmeans.predict(x_test)

plt.figure(figsize=(10,6))
plt.scatter(x_train[:,0],x_train[:,1], c=kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],marker='*',color='k')

# plt.scatter(x_test[:,0],x_test[:,1], c=clusters)
# plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],marker='*',color='k')
