import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from matplotlib import colors as mcolors
from sklearn.metrics import confusion_matrix

np.set_printoptions(precision=4, threshold=np.inf)


plotTypes = ['accuracy', 'misclass']
models = ['deep']
withTest = False

for p in plotTypes:
    for j in models:
        plotType = p
        model = j
        colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

        meanTrain = np.zeros((2, 2000))
        meanVal = np.zeros((2, 2000))
        meanTest = np.zeros((2, 2000))
        for subject in range(1,48):
            if subject == 44 or subject == 45:
                continue
            try:
                subjectData = np.load("single_subjects\{}-trialwise-singleSubjectNum{}-2.5sec-800epoches.npy".format(model, subject))
                print(subjectData.shape)
                if subject ==1 :        
                    if plotType == 'accuracy':
                        plt.plot((1- subjectData[:,2]), color=colors['skyblue'], linestyle='dashed', linewidth=0.5, label='Single subjects train accuracy')
                        plt.plot((1- subjectData[:,3]), color=colors['peachpuff'], linestyle='dashed', linewidth=0.5, label='Single subjects validation accuracy')
                        if withTest == True:
                            plt.plot((1- subjectData[:,5]), color=colors['lightgreen'], linestyle='dashed', linewidth=0.5, label='Single subjects test accuracy')
                    else:
                        plt.plot(subjectData[:,2], color=colors['skyblue'], linestyle='dashed', linewidth=0.5, label='Single subjects train accuracy')
                        plt.plot(subjectData[:,3], color=colors['peachpuff'], linestyle='dashed', linewidth=0.5, label='Single subjects validation accuracy')
                        if withTest == True:
                            plt.plot(subjectData[:,5], color=colors['lightgreen'], linestyle='dashed', linewidth=0.5, label='Single subjects test accuracy')

                else:
                    if plotType == 'accuracy':
                        plt.plot((1- subjectData[:,2]), color=colors['skyblue'], linestyle='dashed', linewidth=0.5)
                        plt.plot((1- subjectData[:,3]), color=colors['peachpuff'], linestyle='dashed', linewidth=0.5)
                        if withTest == True:
                            plt.plot((1- subjectData[:,5]), color=colors['lightgreen'], linestyle='dashed', linewidth=0.5)

                    else:
                        plt.plot(subjectData[:,2], color=colors['skyblue'], linestyle='dashed', linewidth=0.5)
                        plt.plot(subjectData[:,3], color=colors['peachpuff'], linestyle='dashed', linewidth=0.5)
                        if withTest == True:
                            plt.plot(subjectData[:,5], color=colors['lightgreen'], linestyle='dashed', linewidth=0.5)

                for index in range(0, len(subjectData)):
                    if plotType == 'accuracy':
                        meanTrain[0, index] += 1- subjectData[index, 2]
                        meanVal[0, index] += 1- subjectData[index, 3]
                        if withTest == True:
                            meanTest[0, index] += 1- subjectData[index, 5]
                    else:
                        meanTrain[0, index] += subjectData[index, 2]
                        meanVal[0, index] += subjectData[index, 3]
                        if withTest == True:
                            meanTest[0, index] += subjectData[index, 5]
                    meanTrain[1,index] += 1
                    meanVal[1,index] += 1
                    meanTest[1,index] += 1
            except:
                continue

        # print(meanTrain)
        # print(meanVal)
        # print(meanTrain[0]/meanTrain[1])
        if plotType == 'accuracy':
            plt.title('Accuracy rate / epoches')
            plt.ylabel('accuracy')
        else:
            plt.title('Misclass rate / epoches')
            plt.ylabel('misclass')
        plt.xlabel('epoches')
        plt.plot(meanTrain[0]/meanTrain[1], color=colors['darkblue'], linestyle='solid', linewidth=1.5, label='Mean train accuracy')
        plt.plot(meanVal[0]/meanVal[1], color=colors['red'], linestyle='solid', linewidth=1.5, label='Mean validation accuracy')
        if withTest == True:
            plt.plot(meanTest[0]/meanTest[1], color=colors['darkgreen'], linestyle='solid', linewidth=1.5, label='Mean test accuracy')
        plt.legend(loc='best')
        # plt.show()
        if withTest == True:
            plt.savefig("mean-{}-plot-WITH_TEST-{}-cropped-finetune-2.5sec-800epoches.png".format(plotType, model), bbox_inches='tight')
        else:
            plt.savefig("mean-{}-plot-{}-trialwise-finetune-2.5sec-800epoches.png".format(plotType, model), bbox_inches='tight')
        plt.close()
