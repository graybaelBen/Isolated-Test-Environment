# script to automate batch testing
# Runs all batches and writes results in columuns to one csv file
import main
import numpy as np

batches = ["Pristine","B1.2","B1.3","B1.4","B1.5","B2.1","B2.2","B2.3","B2.4","B2.5"]

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