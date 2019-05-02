import fileinput

dataset = ["2",
"data mining",
"frequent pattern mining",
"mining frequent patterns from the transaction dataset",
"closed and maximal pattern mining"]

inVal = []
minsupport = 0
data = []
candidates = []
allFrequent = []
frequent = dict()
sortedFrequent = dict()
dataFrequent = []
count = 0

# 1. Get Input
# for line in fileinput.input():
#     inVal.append(line.rstrip())
inVal = dataset
minsupport = int(inVal[0])
inVal.remove(inVal[0])
for line in inVal:
    data.append(line.split(" "))

# 1.1 Get initial candidates
for entry in data:
    for val in entry:
        if val not in candidates:
            candidates.append(val)
candidates = sorted(candidates)

while True:
    # count for number of elements
    count += 1
    
    # 2. Get Sorted Frequent
    for entryData in data:
        for entryCand in candidates:
            temp = entryCand.split(" ")
            result = all(elem in entryData for elem in temp)
            if result:
                if entryCand in frequent:
                    frequent[entryCand] += 1
                else:
                    frequent[entryCand] = 1

    # 3. Remove Based on MinSupport
    delKey = []
    for key in frequent:
        if frequent[key] < minsupport:
            delKey.append(key)
    for key in delKey:
        del frequent[key]
        candidates.remove(key)
    if count == 1:
        defaultCandidates = frequent.copy()
        
    # 4. Append to AllFrequent
    # allFrequent.update(frequent)  
    temp = list(frequent.items())
    tempFrequent = []
    for val in temp:
        if val not in allFrequent:
            tempFrequent.append(val)
    
    # 4.1 Insert new frequent item into list in the correct order
    if len(allFrequent) == 0:
        allFrequent = tempFrequent
    else:
        for val in tempFrequent:
            allFrequent.append(val)
        
    # 5. Create New Candidates
    newCandidates = []
    for i in range(0, len(candidates)-1):
        strx = candidates[i]
        for j in range(i+1, len(candidates)):
            newCandidates.append(strx + " " + candidates[j])
    # 5.1 Sort candidates and remove duplicates
    newList = []
    for cand in newCandidates:
        temp = cand.split(" ")
        tempList = sorted(list(set(temp)))
        if len(tempList) == count+1 and tempList not in newList:
            newList.append(tempList)
    # 5.2 Turn candidate list to string
    newCand = []
    for cand in newList:
        strx = cand[0]
        for i in range(1, len(cand)):
            strx = strx + " " + cand[i]
        newCand.append(strx)   
    candidates = newCand

    # 6. Break Condition
    if len(candidates) < 1:
        break

allFrequent = sorted(allFrequent)
sortedFrequent = dict(allFrequent)
allFrequent = sorted(sortedFrequent.items(), key=lambda x: x[1], reverse=True)

# print(allFrequent)
# print(defaultCandidates)
for val in allFrequent:
    print(str(val[1]) + " [" + val[0] + "]")
print("")

maxCandidates = []
maxDict = dict()
for val in allFrequent:
    cand = val[0].split(" ")
    for entry in cand:
        if defaultCandidates[entry] == val[1]:
            if entry in maxDict:
                maxCandidates.remove(maxDict[entry])
            maxDict[entry] = val
            if val not in maxCandidates:
                maxCandidates.append(val)
    if len(maxDict) == len(defaultCandidates):
        break
        
# print(maxCandidates)
# print(maxDict)
for val in maxCandidates:
    print(str(val[1]) + " [" + val[0] + "]")
print("")

minCandidates = []
minDict = dict()
for val in maxCandidates:
    cand = val[0].split(" ")
    for entry in cand:
        if entry not in minDict:
            minCandidates.append(val)
        minDict[entry] = val
        
    if len(minDict) == len(defaultCandidates):
        break

minValues = sorted(minDict.items())
minArr = []
for val in minValues:
    if val[1] not in minArr:
        minArr.append(val[1])
# print(minArr)
sortedFrequent = dict(minArr)
minArr = sorted(sortedFrequent.items(), key=lambda x: x[1], reverse=True)
for val in minArr:
    print(str(val[1]) + " [" + val[0] + "]")