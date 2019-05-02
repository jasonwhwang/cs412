import fileinput

dataset = ["good grilled fish sandwich and french fries , but the service is bad",
"disgusting fish sandwich , but good french fries",
"their grilled fish sandwich is the best fish sandwich , but pricy",
"A B A B A B A"]

inVal=[]
data=[]
origPatterns = dict()
allPatterns = dict()
patterns = dict()
newPatterns = dict()
removeP = []
locations = dict()

# 1. Get Data
# for line in fileinput.input():
#     inVal.append(line.rstrip())
inVal = dataset
for line in inVal:
    data.append(line.split(" "))

# 2. Create Dicts for SPADE Algo
for i in range(0, len(data)):
    for j in range(0, len(data[i])):
        if data[i][j] in patterns:
            patterns[data[i][j]].append((i,j))
        else:
            patterns[data[i][j]] = []
            patterns[data[i][j]].append((i,j))
        locations[(i,j)] = data[i][j]

# 3. Prune Dicts to minsupport
for key, val in patterns.items():
    if len(val) < 2:
        removeP.append(key)
        del locations[val[0]]
for r in removeP:
    del patterns[r]
removeP = []

origPatterns.update(patterns)

# 4. Get New Patterns
for i in range (2, 6):
    if len(patterns) <= 0:
        break
    
    for key, val in patterns.items():
        for loc in val:
            nextLoc = (loc[0], loc[1]+1)
            if nextLoc in locations:
                newKey = key + " " + locations[nextLoc]
                if newKey in newPatterns:
                    newPatterns[newKey].append(nextLoc)
                else:
                    newPatterns[newKey] = []
                    newPatterns[newKey].append(nextLoc)

    for key, val in newPatterns.items():
        if len(val) < 2:
            removeP.append(key)
    for r in removeP:
        del newPatterns[r]
    removeP = []

    allPatterns.update(newPatterns)
    patterns = newPatterns
    newPatterns = dict()

newList = dict()
for key, val in allPatterns.items():
    newList[key] = len(val)

newList = list(newList.items())
newList = sorted(newList)
newList = sorted(newList, key=lambda x: x[1], reverse=True)
newList = newList[:20]
for val in newList:
    print("[" + str(val[1]) + ", '" + val[0] + "']")