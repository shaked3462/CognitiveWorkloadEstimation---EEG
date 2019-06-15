import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors


def getAccBeforeFinetune(details):
    if details[0] == 'deep' and details[1] == 'cropped':
        return (1 - 0.250749)
    elif details[0] == 'deep' and details[1] == 'trialwise':
        return (1 - 0.246169)
    elif details[0] == 'shallow' and details[1] == 'cropped':
        return (1 - 0.528161)
    elif details[0] == 'shallow' and details[1] == 'trialwise':
        return (1 - 0.479347)

# plotTypes = ['accuracy'] # plotTypes = ['accuracy', 'misclass']
# models = ['deep', 'shallow']
# trainingType = ['cropped', 'shallow']
# learningStyle = ['finetune', 'single']
def processSinglePlots(learningStyle, trainingType, plotTypes, models):
    np.set_printoptions(precision=4, threshold=np.inf)

    # plotTypes = ['accuracy'] # plotTypes = ['accuracy', 'misclass']
    # models = ['deep', 'shallow']

    for p in plotTypes:
        for j in models:
            for k in trainingType:
                plotType = p
                model = j
                colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

                meanTrain = np.zeros((2, 2000))
                meanVal = np.zeros((2, 2000))
                meanTest = np.zeros((2, 2000))
                for subject in range(1,53):
                    try:
                        if learningStyle == 'finetune':
                            subjectData = np.load("finetuneCrossSubjects\{}-{}-singleSubjectNum{}-2.5sec-800epoches.npy".format(model, k, subject))
                        else: #single
                            subjectData = np.load("logsSingleSubjects\{}-{}-singleSubjectNum{}-2.5sec-800epoches.npy".format(model, k, subject))

                        # print(subjectData.shape)
                        if subject == 1: # different case to create legend only once     
                            if plotType == 'accuracy':
                                plt.plot((1- subjectData[:,2]), color=colors['skyblue'], linestyle='dashed', linewidth=0.5, label='Single subjects train accuracy')
                                plt.plot((1- subjectData[:,3]), color=colors['peachpuff'], linestyle='dashed', linewidth=0.5, label='Single subjects validation accuracy')
                            else:
                                plt.plot(subjectData[:,2], color=colors['skyblue'], linestyle='dashed', linewidth=0.5, label='Single subjects train misclass')
                                plt.plot(subjectData[:,3], color=colors['peachpuff'], linestyle='dashed', linewidth=0.5, label='Single subjects validation misclass')

                        else:
                            if plotType == 'accuracy':
                                plt.plot((1- subjectData[:,2]), color=colors['skyblue'], linestyle='dashed', linewidth=0.5)
                                plt.plot((1- subjectData[:,3]), color=colors['peachpuff'], linestyle='dashed', linewidth=0.5)
                            else:
                                plt.plot(subjectData[:,2], color=colors['skyblue'], linestyle='dashed', linewidth=0.5)
                                plt.plot(subjectData[:,3], color=colors['peachpuff'], linestyle='dashed', linewidth=0.5)

                        for index in range(0, len(subjectData)):
                            if plotType == 'accuracy':
                                meanTrain[0, index] += 1 - subjectData[index, 2]
                                meanVal[0, index] += 1 - subjectData[index, 3]
                            else:
                                meanTrain[0, index] += subjectData[index, 2]
                                meanVal[0, index] += subjectData[index, 3]
                            meanTrain[1,index] += 1
                            meanVal[1,index] += 1
                            meanTest[1,index] += 1
                    except:
                        continue

                if plotType == 'accuracy':
                    plt.title('Accuracy rate / epoches')
                    plt.ylabel('accuracy')
                else:
                    plt.title('Misclass rate / epoches')
                    plt.ylabel('misclass')
                plt.xlabel('epoches')
                plt.plot(meanTrain[0]/meanTrain[1], color=colors['darkblue'], linestyle='solid', linewidth=1.5, label='Mean train accuracy')
                plt.plot(meanVal[0]/meanVal[1], color=colors['red'], linestyle='solid', linewidth=1.5, label='Mean validation accuracy')
                plt.legend(loc='best')
                # plt.show()
                plt.savefig("Plots\mean-{}-plot-{}-{}-finetune-2.5sec-800epoches.png".format(plotType, model, k), bbox_inches='tight')
                plt.close()


# xAxis = ['deep', 'cropped', 'single']
# yAxis = ['shallow', 'trialwise', 'cross']
def processSingleScatterPlots(xAxis, yAxis):
    np.set_printoptions(precision=4, threshold=np.inf)
    plotTypes = 'accuracy'
    # models = ['deep', 'shallow']
    # trainStrategies = ['cropped', 'trialwise']
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

    # xAxis = ['deep', 'cropped', 'single']
    # yAxis = ['shallow', 'trialwise', 'cross']

    accTestX = np.zeros(47)
    accTestY = np.zeros(47)
    j = 0
    for subject in range(1,53):
        bestY = 0
        bestX = 0
        try:
            if xAxis[2] == 'single':
                subjectDataX = np.load("logsSingleSubjects\{}-{}-singleSubjectNum{}-2.5sec-800epoches.npy".format(xAxis[0], xAxis[1], subject))
            else:
                accTestX[j] = 1 - np.load("finetuneCrossSubjects\{}-{}-singleSubjectNum{}-2.5sec-800epoches-testSetMisclass.npy".format(xAxis[0], xAxis[1], subject))

            if yAxis[2] == 'single':
                subjectDataY = np.load("logsSingleSubjects\{}-{}-singleSubjectNum{}-2.5sec-800epoches.npy".format(yAxis[0], yAxis[1], subject))
            else:
                accTestY[j] = 1 - np.load("finetuneCrossSubjects\{}-{}-singleSubjectNum{}-2.5sec-800epoches-testSetMisclass.npy".format(yAxis[0], yAxis[1], subject))                
            
            if xAxis[2] == 'single':
                for i in range(0, len(subjectDataX)):
                    if subjectDataX[i, 4] >= bestX:
                        bestX = subjectDataX[i, 4]
                        accTestX[j] = subjectDataX[i, 5]
            if yAxis[2] == 'single':
                for i in range(0, len(subjectDataY)): 
                    if subjectDataY[i, 4] >= accTestY:
                        accTestY = subjectDataY[i, 4]
                        accTestY[j] = subjectDataY[i, 5]

            j += 1
        except:
            continue

    testMeanX = np.mean(np.trim_zeros(accTestX))*100
    testMeanY = np.mean(np.trim_zeros(accTestY))*100
    plt.title('Accuracy rate / epoches')
    plt.ylabel('Accuracy [%]')
    plt.xlabel('Accuracy [%]')

    plt.plot(range(0, 101), color=colors['black'], linestyle='solid', linewidth=1)
    plt.scatter(np.trim_zeros(accTestX)*100, np.trim_zeros(accTestY)*100, marker='+', color=colors['red'], label='Single subjects accuracy', linestyle='solid', alpha=0.5, linewidth=1.5)
    plt.scatter(testMeanX, testMeanY, marker='o', color=colors['blue'], linestyle='solid', label='Mean accuracy', linewidth=1.5)

    # add accuracy before finetuning
    if yAxis[2] == 'cross' or xAxis[2] == 'cross':
        # initial value is testMean so that if it is single it will compare to curr test accuracy
        testBeforeFinetuneX = testMeanX
        testBeforeFinetuneY = testMeanY
        if xAxis[2] == 'cross':
            # accuracy values are from the final epoch in the log file
            testBeforeFinetuneX = getAccBeforeFinetune(xAxis)*100
            print('test acc for x axis before finetune is {}'.format(testBeforeFinetuneX))
        if yAxis[2] == 'cross':
            # accuracy values are from the final epoch in the log file
            testBeforeFinetuneY = getAccBeforeFinetune(yAxis)*100
            print('test acc for y axis before finetune is {}'.format(testBeforeFinetuneY))

        plt.scatter(testBeforeFinetuneX, testBeforeFinetuneY, marker='o', color=colors['green'], linestyle='solid', label='Accuracy before finetune', linewidth=1.5)
    print("X: {}".format(xAxis))
    print("Y: {}".format(yAxis))
    print("Mean accuracies: X = {}, Y = {}".format(testMeanX, testMeanY))
    plt.legend(loc='best')
    plt.show()
    # plt.savefig("Plots\scatterPlot-X-{}-Y-{}.png".format(xAxis, yAxis), bbox_inches='tight')
    plt.close()

# processSinglePlots('finetune', ['cropped', 'trialwise'], ['accuracy'], ['deep', 'shallow'])

for yAxis in [['deep', 'cropped', 'cross'], ['deep', 'trialwise', 'cross']]:
    for xAxis in [['shallow', 'cropped', 'cross'], ['shallow', 'trialwise', 'cross']]:
        # print('xAxis')
        # print(xAxis)
        # print('yAxis')
        # print(yAxis)
        processSingleScatterPlots(xAxis, yAxis)