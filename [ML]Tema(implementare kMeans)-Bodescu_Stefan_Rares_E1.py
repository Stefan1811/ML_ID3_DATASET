import pandas as pd
import numpy as np
import math as mt


train_data = pd.read_csv('heart_attack.csv')
train_data = train_data.dropna()

mean_values = train_data.mean()
variance_values = train_data.var()

#print("Mean values:\n", mean_values)
#print("\nVariance values:\n", variance_values)

target_attribute = 'target'
discrete_attributes = ['sex','cp','fbs','restecg','exang','slope','ca','thal']
continuous_attributes=['age','trestbps','chol','thalach','oldpeak']

train_data = train_data.drop(target_attribute, axis=1)

def distance_points(a,b,p):
    distance=0
    if len(a)!= len(b):
        return -1
    else:
        for i in range(len(a)):
            distance+=abs(a[i]-b[i])**p
        return distance**(1/p)

a=[1,2,3]
b=[4,5,6]
#print(" Minkowski distance:\n",distance_points(a,b,2))

def generate_random_points(n,dimensions,left_range, right_range):
    points=[]
    for i in range(n):
        point=[]
        for j in range(dimensions):
            point.append(np.random.uniform(left_range,right_range))
        points.append(point)
    return points

#print("Random points:\n",generate_random_points(5,3,0,10))

def distance_to_df(train_data,x,p):
    if len(x) != len(train_data.columns):
        raise ValueError("Point X must have the same number of dimensions as DataFrame df")

    distances = np.linalg.norm(df.values - np.array(x), ord=p, axis=1)
    return distances

data = {'Feature1': [1, 2, 3],
        'Feature2': [2, 5, 1],
        'Feature3': [3, 6, 2]}
df = pd.DataFrame(train_data)


point_X = [1, 2, 3]

#print("Distance to df:\n",distance_to_df(df,point_X,2))

def distance_to_centroids(train_data,centroids_list,p):
    distances=[]
    for centroid in centroids_list:
           distances.append(distance_to_df(train_data,centroid,p))
    return distances

centroids_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print("Distance to centroids:\n",distance_to_centroids(df,centroids_list,2))

def closest_centroid(distances):
    closest_centroid=[]
    for i in range(len(distances[0])):
        closest_centroid.append(np.argmin([distances[j][i] for j in range(len(distances))]))
    return closest_centroid

distances=distance_to_centroids(df,centroids_list,2)
print("Closest centroid:\n",closest_centroid(distances))

def get_clusters(closest_centroid_list):
    clusters = {}
    
    for i, centroid_index in enumerate(closest_centroid_list):
        if centroid_index not in clusters:
            clusters[centroid_index] = []
        clusters[centroid_index].append(i)
    
    for i in range(len(closest_centroid_list)):
        if i not in clusters:
            clusters[i] = []
    
    return clusters


closest_centroid_list=closest_centroid(distances)
print("Clusters:\n",get_clusters(closest_centroid_list))

def update_centroids(train_data, centroids_list, clusters):
    new_centroids = []
    for centroid_index in range(len(clusters)):
        point_indices = clusters.get(centroid_index, [])
        if not point_indices:
            new_centroids.append(centroids_list[centroid_index])
        elif len(point_indices) == 1:
            new_centroids.append(centroids_list[centroid_index])
        else:
            cluster_points = train_data.iloc[point_indices]
            new_centroid = cluster_points.mean().values
            new_centroids.append(new_centroid)
    return new_centroids


clusters=get_clusters(closest_centroid_list)
print("New centroids:\n",update_centroids(df,centroids_list,clusters))

def kmeans_plusplus_init(train_data, n_clusters, random_seed=None):
    np.random.seed(random_seed)
    centroids = [train_data.iloc[np.random.choice(len(df))].values]
    for _ in range(1, n_clusters):
        distances = np.array([min(np.linalg.norm(point - centroid) ** 2 for centroid in centroids) for point in train_data.values])
        probabilities = distances / distances.sum()
        new_centroid = train_data.iloc[np.random.choice(len(df), p=probabilities)].values
        centroids.append(new_centroid)
    return centroids

print("Kmeans++:\n",kmeans_plusplus_init(df,3,random_seed=42))

def kmeans(train_data, n_clusters, max_iter=100, init_type='random', random_seed=None):
    if init_type == 'random':
        centroids = generate_random_points(n_clusters, len(train_data.columns), 0, 10)
    elif init_type == 'kmeans++':
        centroids = kmeans_plusplus_init(train_data, n_clusters, random_seed)
    else:
        raise ValueError('Unsupported init_type: %s' % init_type)
    
    for _ in range(max_iter):
        distances = distance_to_centroids(train_data, centroids, 2)
        closest_centroids = closest_centroid(distances)
        clusters = get_clusters(closest_centroids)
        new_centroids = update_centroids(train_data, centroids, clusters)
        
        if np.allclose(centroids, new_centroids):
            break
        
        centroids = new_centroids
    
    result = {'clusters': clusters, 'centroids': centroids}
    return result

print("Kmeans:\n",kmeans(df,3,random_seed=42))

def calculate_J_score(df, membership, centroids):
    J_score = 0
    for i, centroid in enumerate(centroids):
        cluster_points = df.iloc[membership[i]]
        J_score += np.linalg.norm(cluster_points - centroid) ** 2
    return J_score

print("J score:\n",calculate_J_score(df,clusters,centroids_list))
