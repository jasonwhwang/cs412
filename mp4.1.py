import fileinput
from math import sqrt, inf

def k_means_init(data):
    N = int(data[0][0])
    k = int(data[0][1])
    stop = len(data) - k
    centroids = data[stop:]
    points = data[1:stop]
    return N, k, centroids, points

def k_means_distance(centeroid, point):
    distance = 0.0
    for i in range(len(centeroid)):
        distance += (centeroid[i]-point[i])**2
    return sqrt(distance)

def k_means_clusters(centeroids, points):
    clusters = [[] for centeroid in centeroids]
    for point in points:
        index = 0
        distance = inf
        for i in range(len(centeroids)):
            new_distance = k_means_distance(centeroids[i], point)
            if new_distance < distance:
                index = i
                distance = new_distance
        clusters[index].append(point)
    return clusters

def k_means_centeroids(clusters):
    centeroids = []
    for cluster in clusters:
        centeroid = [sum(row[i] for row in cluster) for i in range(len(cluster[0]))]
        centeroid = [x/len(cluster) for x in centeroid]
        centeroids.append(centeroid)
    return centeroids


dataset = ["10 2",
           "8.98320053625 -2.08946304844",
           "2.61615632899 9.46426282022",
           "1.60822068547 8.29785986996",
           "8.64957587261 -0.882595891607",
           "1.01364234605 10.0300852081",
           "1.49172651098 8.68816850944",
           "7.95531802235 -1.96381815529",
           "0.527763520075 9.22731148332",
           "6.91660822453 -3.2344537134",
           "6.48286208351 -0.605353440895",
           "3.35228193353 6.27493570626",
           "6.76656276363 6.54028732984"]

d, data = [], []
d = dataset

# Get Data
# for line in fileinput.input():
#     d.append(line.rstrip())
for line in d:
    data.append([float(i) for i in line.split(" ")])

# K-Means Clustering
# 1. Initialization
N, k, centeroids, points = k_means_init(data)
clusters = None

while True:
    # 2. Get New Clusters
    clusters = k_means_clusters(centeroids, points)
    # 3. Get New Centroids
    new_centeroids = k_means_centeroids(clusters)
    if new_centeroids == centeroids:
        break
    centeroids = new_centeroids

for point in points:
    for i in range(len(clusters)):
        if point in clusters[i]:
            print(i)