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

accTestShallow = np.zeros(47)
accTestDeep = np.zeros(47)
j = 0
for subject in range(1,53):
    bestDeep = 0
    bestShallow = 0
    try:
        subjectDataShallow = np.load("single_subjects\shallow-trialwise-singleSubjectNum{}-2.5sec-800epoches.npy".format(subject))
        subjectDataDeep = np.load("single_subjects\deep-trialwise-singleSubjectNum{}-2.5sec-800epoches.npy".format(subject))
        for i in range(0, len(subjectDataShallow)):
            if subjectDataShallow[i, 4] >= bestShallow:
                bestShallow = subjectDataShallow[i, 4]
                accTestShallow[j] = subjectDataShallow[i, 5]
        for i in range(0, len(subjectDataDeep)):
            if subjectDataDeep[i, 4] >= bestDeep:
                bestDeep = subjectDataDeep[i, 4]
                accTestDeep[j] = subjectDataDeep[i, 5]
        j += 1
    except:
        continue

testMeanShallow = np.mean(accTestShallow)*100
testMeanDeep = np.mean(accTestDeep)*100
# plt.title('Accuracy rate / epoches')
plt.ylabel('Accuracy [%]')
plt.xlabel('Accuracy [%]')
plt.plot(range(0, 101), color=colors['black'], linestyle='solid', linewidth=1)
plt.scatter(accTestShallow*100, accTestDeep*100, marker='+', color=colors['red'], label='Single subjects accuracy', linestyle='solid', alpha=0.5, linewidth=1.5)
plt.scatter(testMeanShallow, testMeanDeep, marker='o', color=colors['blue'], linestyle='solid', label='Mean accuracy', linewidth=1.5)
plt.legend(loc='best')
plt.show()
# if withTest == True:
#     plt.savefig("mean-{}-plot-WITH_TEST-{}-trialwise-single_subjects-2.5sec-800epoches.png".format(plotType, model), bbox_inches='tight')
# else:
#     plt.savefig("mean-{}-plot-{}-trialwise-single_subjects-2.5sec-800epoches.png".format(plotType, model), bbox_inches='tight')
# plt.close()
