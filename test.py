# script to automate batch testing
# Runs all batches and writes results in columuns to one csv file
import clean_slate
import numpy as np
import main

batches = ["Pristine","Batch1.1","Batch1.2","Batch1.3","Batch1.4","Batch1.5","Batch2.1","Batch2.2","Batch2.3","Batch2.4","Batch2.5"]

resultMatrix = []
longest = 0

for batch in batches:
    matchCountArray, kpCountArray = main.run(batch)
    matchCountArray.insert(0,f"{batch} Match Counts")
    kpCountArray.insert(0,f"{batch} Keypoint Counts")
    if len(matchCountArray) > longest: longest = len(matchCountArray)
    resultMatrix.append(matchCountArray)
    resultMatrix.append(kpCountArray)

# fill rows with ''; must be a equal sized matrix to transpose
for row in resultMatrix:
    row.extend(['']*(longest-len(row)))

np.savetxt('results.csv', np.transpose(resultMatrix), delimiter=',',fmt='%s')
print('done!')