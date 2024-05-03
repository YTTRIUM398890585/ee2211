import numpy as np
import matplotlib.pyplot as plt

""" 3D data points and centers for testing the k-means algorithm """
# # Data points
# x1 = np.array([0, 0, 1])
# x2 = np.array([0, 1, 1])
# x3 = np.array([1, 1, 1])
# x4 = np.array([1, 0, 2])
# x5 = np.array([3, 0, 2])
# x6 = np.array([3, 1, 2])
# x7 = np.array([4, 0, 2])
# x8 = np.array([4, 1, 1])
# data_points = np.array ([x1, x2, x3, x4, x5, x6 , x7 , x8])

# # Initial centers
# c1_init = np.array([0, 0, 4])
# c2_init = np.array([3, 0, 5])
# centers = np.array([c1_init , c2_init])

""" 2D data points and centers for testing the k-means algorithm """
# Data points
x = np.array([1, 1])
y = np.array([0, 1])
z = np.array([0, 0])
data_points = np.array([x, y, z])

# Initial centers
centers = np.array([x, y])

""" 1D data points and centers for testing the k-means algorithm """
# # Data points
# data_points = np.array([[10], [11], [12], [15], [16], [17], [20], [21], [22]])

# # Initial centers
# centers = np.array([[10], [15], [20]])

''' Q 34'''
# Data points
data_points = np.array([[10.0], [11.0], [14.0], [16.0], [16.0], [18.0], [18.0], [21.0], [21.0]])

# Initial centers
centers = np.array([[10], [16], [21]])

 

""" random 2D data points and centers for testing the k-means algorithm """
# # Set three centers , the model should predict similar results
# center_1 = np.array([2, 2])
# center_2 = np.array([4, 4])
# center_3 = np.array([6, 1])

# # Generate random data and center it to the three centers
# data_1 = np.random.randn(200, 2) + center_1
# data_2 = np.random.randn(200, 2) + center_2
# data_3 = np.random.randn(200, 2) + center_3
# data_points = np.concatenate((data_1, data_2, data_3), axis = 0)

# # initialize cluster centers
# k=2
# centers = data_points[np.random.choice(len(data_points), k, replace=False)]

""" k-means algorithm """
def k_means(data_points, centers, n_clusters, max_iterations=100, tol=1e-4):
    for _ in range (max_iterations):
        # Assign each data point to the closest centroid
        labels = np.argmin(np.linalg.norm(data_points[:, np.newaxis] - centers , axis=2), axis=1)

        # Update centroids to be the mean of the data pointsassigned to them
        new_centers = np.zeros((n_clusters , data_points.shape[1]))

        # End if centroids no longer change
        for i in range (n_clusters):
            # If no data points are assigned to the i-th cluster
            if not np.any(labels == i):
                print("[WARNING] no data points assigned to cluster ", i)
                
            new_centers[i] = data_points[labels == i].mean(axis=0)
            
        if np.linalg.norm(new_centers - centers) < tol:
            break
        
        centers = new_centers
        
    return centers , labels

""" results """
centers, labels = k_means(data_points, centers, n_clusters=len(centers))
print ("Converged centers :", centers)
print ("Converged labels :", labels)

""" plotting the clustering results only for 2D data """
plt.title("Clustering Results (only make sense for 2D data)")
plt.scatter(data_points[:, 0], data_points[:, 1], c=labels , cmap="viridis", alpha=0.5)
plt.scatter(centers[:, 0], centers[:, 1], marker="*", s=200, c="k")
plt.show()