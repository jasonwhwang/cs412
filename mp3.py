import fileinput
from math import sqrt

##### Decision Tree Functions #####

# Function to take all rows of the data and split it into two datasets
# based on the attribute and the attribute threshold score.
def split_data(attributeVal, attributeScore, data):
    list1, list2 = [], []

    for row in data:
        if row[attributeVal] <= attributeScore:
            list1.append(row)
        else:
            list2.append(row)
    return list1, list2


# Calculate the Gini Score of a Split Dataset as defined in the lecture slides
# 𝑔𝑖𝑛𝑖(𝐷) = (class1 proportion) * (1 − sum(probabilitiesClass1^2))
#          + (class2 proportion) * (1 − sum(probabilitiesClass2^2))
def gini(splits, classes):
    gini = 0.0
    total = float(sum([len(s) for s in splits]))

    for split in splits:
        p_2 = 0.0
        num = float(len(split))
        if num == 0:
            continue
            
        # For each of the classes, calculate their proportions
        for c in classes:
            p = [row[-1] for row in split].count(c) / num
            p_2 += p*p
        gini += (num/total) * (1.0-p_2)
        
    return gini



# Get the best split by choosing the best attribute and attribute threshold
def best_split_data(data):
    classes = list(set(row[-1] for row in data))
    giniScores = []

    # For all rows of data, and for each attribute
    # split the data based on that attribute
    for row in data:
        for attribute in range(len(row)-1):
            splits = split_data(attribute, row[attribute], data)
            g = gini(splits, classes)
            giniScores.append((attribute, row[attribute], g, splits))

    # Sort the Gini Scores by attribute and choose the smallest gini
    giniScores = sorted(giniScores, key=lambda x: x[2])
    return giniScores[0][0], giniScores[0][1], giniScores[0][2], giniScores[0][3]



# Class Definition for the Binary Decision Tree
class Node:
    def __init__(self, attributeVal, attributeSplit, leftChild, rightChild, predict):
        self.attributeVal = attributeVal
        self.attributeSplit = attributeSplit
        self.leftChild = leftChild
        self.rightChild = rightChild
        self.predict = predict

        
# Build a decision tree recursively
# Create new nodes until max depth is reached or no splitting is necessary
# ie. only one list after split
def decision_tree(data, depth, max_depth):
    attributeVal, attributeSplit, attributeScore, splitLists = best_split_data(data)
    newNode = None

    if depth < max_depth and len(splitLists[0]) != 0 and len(splitLists[1]) != 0:
        # Branch Nodes
        leftNode = decision_tree(splitLists[0], depth+1, max_depth)
        rightNode = decision_tree(splitLists[1], depth+1, max_depth)
        newNode = Node(attributeVal, attributeSplit, leftNode, rightNode, None)

    else:
        # Leaf Nodes
        classes = list(set(row[-1] for row in data))
        preDict = dict()
        for c in classes:
            preDict[c] = [row[-1] for row in data].count(c)
        
        # Choose the prediction based on the most numbered class
        # Sorted to account for tie-breaker
        preDict = {k: preDict[k] for k in sorted(preDict)}
        predict = max(preDict, key=preDict.get)
        newNode = Node(None, None, None, None, predict)

    return newNode

# Make a prediction using a decision tree by traversing tree
# by comparing to the threshold values of each node.
def predict_DT(data, node):
    if node.attributeVal != None:
        if data[node.attributeVal] <= node.attributeSplit:
            return predict_DT(data, node.leftChild)
        else:
            return predict_DT(data, node.rightChild)
    else:
        return node.predict



##### KNN Functions #####
# Euclidean Distance is the square root of the total squared difference
# of the attribute values
def euclidean(value, comparison):
    d = 0
    for i in range(len(value)-1):
        d += pow((value[i] - comparison[i]), 2)
    return sqrt(d)

# Closest Neighbors are calculated by finding the difference between
# a data point and all of the other data point values.
# The class is then choosen by finding the closest classes
def neighbors(value, allValues, k):
    distances = []

    # Find the distance between a value and all other values
    for i in range(len(allValues)):
        dist = euclidean(value, allValues[i])
        distances.append((dist, allValues[i]))
    distances = sorted(distances, key=lambda x: x[0])

    # Return only the first k neighbors
    neighbors = []
    for i in range(k):
        neighbors.append(distances[i][1])
    return neighbors



# Predict using K-Nearest Neighbors
def predict_KNN(data):
    predictedClasses = dict()
    
    # Count all of the neighbors and their classes
    # Use Majority Voting to determine what the class should be
    for i in range(len(data)):
        prediction = data[i][-1]
        if prediction not in predictedClasses:
            predictedClasses[prediction] = 1
        else:
            predictedClasses[prediction] += 1

    # Sort the classes and their counts and choose the highest one
    # Tie breaker is chosen with the smallest label
    predictedClasses = {k: predictedClasses[k] for k in sorted(predictedClasses)}
    predict = max(predictedClasses, key=predictedClasses.get)
    return predict


##### Data Cleaning & Spliting #####
# This function will remove attribute labels, sort them by order, and place class label at the end
# ie. data = [ [attribute0Val, attribute1Val, attribute2Val, class] ]
def train_test_data(data):
    train, test = [], []

    for line in data:
        tempList = []
        tempDict = dict()
        
        # Split the attributes and add them to a dictionary
        for i in range(1, len(line)):
            entry = line[i].split(":")
            tempDict[entry[0]] = entry[1]

        # Sort the dictionary by key value for all datapoints
        tempDict = {k: tempDict[k] for k in sorted(tempDict)}
        for key, val in tempDict.items():
            tempList.append(float(val))
        tempList.append(int(line[0]))

        # Split the data into train and test sets
        if int(line[0]) != 0:
            train.append(tempList)
        else:
            test.append(tempList)
    return train, test


##### Main Function #####
dataset1 = ["1 0:1.0 2:1.0",
            "1 0:1.0 2:2.0",
            "1 0:2.0 2:1.0",
            "3 0:2.0 2:2.0",
            "1 0:3.0 2:1.0",
            "3 0:3.0 2:2.0",
            "3 0:3.0 2:3.0",
            "3 0:4.5 2:3.0",

            "0 0:1.0 2:2.2",
            "0 0:4.5 2:1.0"]

dataset2 = ["1 0:1.0 2:1.0",
            "2 0:1.0 2:2.0",
            "1 0:2.0 2:1.0",
            "3 0:2.0 2:2.0",
            "1 0:3.0 2:1.0",
            "3 0:3.0 2:2.0",
            "3 0:3.0 2:3.0",
            "3 0:4.5 2:3.0",
            "1 0:2.0 2:1.0",
            "2 0:3.0 2:2.0",
            "3 0:1.0 2:5.0",
            "3 0:3.0 2:1.0",
            "1 0:2.0 2:1.0",
            "1 0:4.0 2:1.0",
            "3 0:4.0 2:1.0",
            "3 0:3.0 2:1.0",
            "2 0:6.0 2:5.0",
            "2 0:2.0 2:1.0",
            "1 0:4.0 2:1.0",
            "3 0:3.0 2:1.0",
            "3 0:1.0 2:3.0",
            "0 0:1.0 2:2.2",
            "0 0:4.5 2:1.0",
            "0 0:1.5 2:2.7",
            "0 0:3.2 2:1.2",
            "0 0:1.1 2:2.9",
            "0 0:3.5 2:4.0"]

dataset3 = ["1 0:1.5 2:7.1 5:8.1 4:1.1 9:5.1 3:7.8",
            "1 0:7.0 2:1.1 5:5.1 4:0.1 9:2.1 3:1.2",
            "1 0:1.0 2:1.1 5:8.1 4:6.1 9:7.1 3:8.4",
            "2 0:5.0 2:1.1 5:8.1 4:5.1 9:8.1 3:7.2",
            "1 0:1.0 2:1.1 5:8.1 4:6.1 9:7.1 3:8.4",
            "8 0:2.0 2:2.1 5:2.1 4:5.1 9:8.1 3:7.2",
            "8 0:2.0 2:1.1 5:8.1 4:5.1 9:7.1 3:8.4",
            "8 0:5.0 2:1.1 5:5.1 4:5.1 9:7.1 3:7.2",
            "2 0:1.0 2:4.1 5:8.1 4:2.4 9:4.1 3:11.2",
            "2 0:1.0 2:8.1 5:8.1 4:3.1 9:6.1 3:20.2",
            "3 0:1.0 2:1.1 5:4.1 4:2.5 9:8.6 3:4.2",
            "3 0:1.0 2:1.1 5:1.1 4:8.1 9:7.7 3:2.2",
            "0 0:7.0 2:2.2 5:8.1 4:2.2 9:2.5 3:1.2",
            "0 0:4.5 2:5.0 5:2.5 4:7.1 9:6.8 3:10.2",
            "0 0:0.5 2:1.0 5:1.5 4:3.1 9:7.3 3:2.2",
            "0 0:1.5 2:1.0 5:4.5 4:8.1 9:3.6 3:7.2",
            "0 0:7.0 2:7.2 5:8.1 4:2.2 9:2.5 3:1.2",
            "0 0:4.5 2:5.0 5:8.5 4:7.1 9:7.8 3:5.2",
            "0 0:0.5 2:2.0 5:7.5 4:3.1 9:4.3 3:2.2",
            "0 0:1.5 2:1.0 5:4.5 4:8.1 9:3.6 3:2.2"]

d, data = [], []
d = dataset3

# 1. Get Data
# Save to List
# for line in fileinput.input():
#     d.append(line.rstrip())
for line in d:
    data.append(line.split(" "))

# 2. Split data into Train and Test sets based on class label (if label == 0)
train, test = train_test_data(data)

# 3. Decision Tree Predictions
# Build decision tree and return root node
max_depth = 2
root = decision_tree(train, 0, max_depth)
predictions = []

# For each row in the test dataset, make a prediction 
for row in test:
    predictions.append(predict_DT(row, root))
for p in predictions:
    print(p)
print()

# 4. KNN Predictions
k = 3
predictions = []
# For each row in the test dataset, make a prediction
for row in test:
    # Make a prediction based on the points that it is closest to using the train dataset
    n = neighbors(row, train, k)
    predict = predict_KNN(n)
    predictions.append(predict)
for p in predictions:
    print(p)