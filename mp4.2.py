import fileinput
from math import sqrt, inf
import numpy as np

def agnes_init(data):
    N = int(data[0][0])
    k = int(data[0][1])
    stop = len(data) - k
    centroids = data[stop:]
    points = data[1:stop]
    return N, k, centroids, points

def agnes_distance(point1, point2):
    distance = 0.0
    for i in range(len(point1)):
        distance += (point1[i]-point2[i])**2
    return sqrt(distance)

def agnes_min_idx(matrix):
    minval = inf
    idx1, idx2 = 0, 0
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if matrix[i][j] is not 0 and matrix[i][j] < minval:
                minval = matrix[i][j]
                idx1 = i
                idx2 = j
    return min(idx1, idx2), max(idx1, idx2)

def agnes_modify_matrix(matrix, cluster1, cluster2):
    matrix[cluster1][cluster2] = 0

    for i in range(len(matrix)):
        matrix[i][cluster1] = min(matrix[i][cluster1], matrix[i][cluster2])
        matrix[cluster1][i] = min(matrix[cluster1][i], matrix[cluster2][i])
    return matrix


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
N, k, centeroids, points = agnes_init(data)
clusters = [i for i in range(len(points))]
# 2. Create Distance Matrix
matrix = [[0 for x in range(len(points))] for y in range(len(points))]
# matrix = np.zeros((len(points), len(points)))

for i in range(len(points)):
    for j in range(i, len(points)):
        matrix[j][i] = agnes_distance(points[i], points[j])

while len(set(clusters)) > k:
    # 3. Get Smallest Distance
    # minval = np.min(matrix[np.nonzero(matrix)])
    # minidx = np.where(matrix==minval)
    # min_cluster = min(minidx[0][0], minidx[1][0])
    # max_cluster = max(minidx[0][0], minidx[1][0])
    min_cluster, max_cluster = agnes_min_idx(matrix)

    # 4. Assign Clusters
    minC = min(clusters[min_cluster], clusters[max_cluster])
    maxC = max(clusters[min_cluster], clusters[max_cluster])
    for i in range(len(clusters)):
        if clusters[i] == maxC:
            clusters[i] = minC
    # 5. Modify Matrix Values
    matrix = agnes_modify_matrix(matrix, min_cluster, max_cluster)

for cluster in clusters:
    print(cluster)
