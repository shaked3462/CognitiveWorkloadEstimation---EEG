import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from matplotlib import colors as mcolors
from sklearn.metrics import confusion_matrix

np.set_printoptions(precision=4, threshold=np.inf)


plotTypes = 'accuracy'
# models = ['deep', 'shallow']

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

trainStrategies = ['cropped', 'trialwise']

# for shallowStrategy in trainStrategies:
#     for deepStrategy in trainStrategies:
accTestShallow = np.zeros(40)
accTestDeep = np.zeros(40)
j = 0
for subject in range(1,48):
    if subject == 44 or subject == 45:
        continue
    bestDeep = 0
    bestShallow = 0
    try:
        print("1")
        subjectDataShallow = np.load("single_subjects\deep-cropped-singleSubjectNum{}-2.5sec-800epoches.npy".format(subject))
        print("2")
        print(accTestShallow[j])    
        accTestDeep[j] = 1- np.load("finetuning\deep-trialwise-singleSubjectNum{}-2.5sec-800epoches-testSetMisclass.npy".format(subject))
        print("3")
        print(accTestDeep[j])
        for i in range(0, len(subjectDataShallow)):
            if subjectDataShallow[i, 4] >= bestShallow:
                bestShallow = subjectDataShallow[i, 4]
                accTestShallow[j] = subjectDataShallow[i, 5]
                print(accTestShallow[j])
        # for i in range(0, len(subjectDataDeep)):
        #     if subjectDataDeep[i, 4] >= bestDeep:
        #         bestDeep = subjectDataDeep[i, 4]
        #         accTestDeep[j] = subjectDataDeep[i, 5]
        j += 1
    except:
        continue

testMeanShallow = np.mean(np.trim_zeros(accTestShallow))*100
testMeanDeep = np.mean(np.trim_zeros(accTestDeep))*100
# plt.title('Accuracy rate / epoches')
plt.ylabel('Accuracy [%]')
plt.xlabel('Accuracy [%]')

plt.plot(range(0, 101), color=colors['black'], linestyle='solid', linewidth=1)
plt.scatter(np.trim_zeros(accTestShallow)*100, np.trim_zeros(accTestDeep)*100, marker='+', color=colors['red'], label='Single subjects accuracy', linestyle='solid', alpha=0.5, linewidth=1.5)
plt.scatter(testMeanShallow, testMeanDeep, marker='o', color=colors['blue'], linestyle='solid', label='Mean accuracy', linewidth=1.5)
# print("Training Strategies: Shallow = {}, Deep = {}".format(shallowStrategy, deepStrategy))
print("Mean accuracies: Shallow = {}, Deep = {}".format(testMeanShallow, testMeanDeep))
plt.legend(loc='best')
plt.show()
# if withTest == True:
# plt.savefig("scatterPlot-shallow_single_{}-deep_single_{}.png".format(shallowStrategy, deepStrategy), bbox_inches='tight')
plt.close()
# else:
#     plt.savefig("mean-{}-plot-{}-cropped-single_subjects-2.5sec-800epoches.png".format(plotType, model), bbox_inches='tight')
# plt.close()
