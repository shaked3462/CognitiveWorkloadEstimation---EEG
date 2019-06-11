import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from matplotlib import colors as mcolors
from sklearn.metrics import confusion_matrix

np.set_printoptions(precision=4, threshold=np.inf)


plotTypes = 'accuracy'
# models = ['deep', 'shallow']
# trainStrategies = ['cropped', 'trialwise']
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

xAxis = ['deep', 'cropped', 'single']
yAxis = ['shallow', 'trialwise', 'cross']

# for shallowStrategy in trainStrategies:
#     for deepStrategy in trainStrategies:
accTestX = np.zeros(47)
accTestY = np.zeros(47)
j = 0
for subject in range(1,53):
    bestDeep = 0
    bestShallow = 0
    try:
        if xAxis[2] == 'single':
            subjectDataX = np.load("logsSingleSubjects\{}-{}-singleSubjectNum{}-2.5sec-800epoches.npy".format(xAxis[0], xAxis[1], subject))
        else:
            accTestX[j] = 1 - np.load("finetuneCrossSubjects\{}-{}-singleSubjectNum{}-2.5sec-800epoches-testSetMisclass.npy".format(xAxis[0], xAxis[1], subject))

        if yAxis[2] == 'single':
            subjectDataY = np.load("logsSingleSubjects\{}-{}-singleSubjectNum{}-2.5sec-800epoches.npy".format(xAxis[0], xAxis[1], subject))
        else:
            accTestDeep[j] = 1 - np.load("finetuneCrossSubjects\{}-{}-singleSubjectNum{}-2.5sec-800epoches-testSetMisclass.npy".format(xAxis[0], xAxis[1], subject))                
        
        if xAxis[2] == 'single':
            for i in range(0, len(subjectDataX)):
                if subjectDataX[i, 4] >= bestShallow:
                    bestShallow = subjectDataX[i, 4]
                    accTestShallow[j] = subjectDataX[i, 5]
                    print(accTestShallow[j])
        if yAxis[2] == 'single':
            for i in range(0, len(subjectDataY)): 
                if subjectDataY[i, 4] >= bestDeep:
                    bestDeep = subjectDataY[i, 4]
                    accTestDeep[j] = subjectDataY[i, 5]

        j += 1
    except:
        continue

testMeanShallow = np.mean(np.trim_zeros(accTestShallow))*100
testMeanDeep = np.mean(np.trim_zeros(accTestDeep))*100
plt.title('Accuracy rate / epoches')
plt.ylabel('Accuracy [%]')
plt.xlabel('Accuracy [%]')

plt.plot(range(0, 101), color=colors['black'], linestyle='solid', linewidth=1)
plt.scatter(np.trim_zeros(accTestShallow)*100, np.trim_zeros(accTestDeep)*100, marker='+', color=colors['red'], label='Single subjects accuracy', linestyle='solid', alpha=0.5, linewidth=1.5)
plt.scatter(testMeanShallow, testMeanDeep, marker='o', color=colors['blue'], linestyle='solid', label='Mean accuracy', linewidth=1.5)
print("Training Strategies: Shallow = {}, Deep = {}".format(shallowStrategy, deepStrategy))
print("Mean accuracies: Shallow = {}, Deep = {}".format(testMeanShallow, testMeanDeep))
plt.legend(loc='best')
plt.show()
# plt.savefig("scatterPlot-shallow_{}-deep__{}.png".format(shallowStrategy, deepStrategy), bbox_inches='tight')
plt.close()
