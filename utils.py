import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from sklearn.metrics import confusion_matrix

def loadSubjects(subjectNum):
    for subject_id in range(1,subjectNum+1):
        labelCounters = np.zeros(3) #[0counter, 1counter, 2counter]
        if (subject_id == 15) or (subject_id == 24) or (subject_id == 25) or (subject_id == 34) or (subject_id == 40): #subjects that were excluded from experiment.
            continue

        print("loading data for subject {}".format(subject_id))
        Xtmp = np.load("NirDataset\\croppedDataForSubject{}.npy".format(subject_id))
        ytmp = np.load("NirDataset\\croppedLabelsForSubject{}.npy".format(subject_id))

        for i in range(0,len(ytmp)):
            labelCounters[ytmp[i]] += 1

        labelCounters = labelCounters.astype(int)
        print("INFO : label examples: {}".format(labelCounters))
        examplesPerSubjecti = np.amin(labelCounters)
        print("INFO : used {} examples from each label".format(examplesPerSubjecti))

        Xtmp0, Xtmp1, Xtmp2 = np.split(Xtmp, [labelCounters[0], labelCounters[0]+labelCounters[1]])
        ytmp0, ytmp1, ytmp2 = np.split(ytmp, [labelCounters[0], labelCounters[0]+labelCounters[1]])
        
        X0, y0 = shuffle(Xtmp0, ytmp0)
        X1, y1 = shuffle(Xtmp1, ytmp1)
        X2, y2 = shuffle(Xtmp2, ytmp2)

        examplesPerAllSubjects = np.amin([len(X0), len(X1), len(X2)])
        print("num of examples for each label {}".format(examplesPerAllSubjects))
        X = np.concatenate((X0[:examplesPerAllSubjects], X1[:examplesPerAllSubjects], X2[:examplesPerAllSubjects]), axis=0)
        y = np.concatenate((y0[:examplesPerAllSubjects], y1[:examplesPerAllSubjects], y2[:examplesPerAllSubjects]), axis=0)
        X, y = shuffle(X, y)

        print(X.shape)
        sizeOf70PercentOfDataSet = int(X.shape[0]*7/10)

        np.save("NirDataset\Single\subject{}_data_train".format(subject_id), X[:sizeOf70PercentOfDataSet])
        np.save("NirDataset\Single\subject{}_data_test".format(subject_id), X[sizeOf70PercentOfDataSet:])
        np.save("NirDataset\Single\subject{}_labels_train".format(subject_id), y[:sizeOf70PercentOfDataSet])
        np.save("NirDataset\Single\subject{}_labels".format(subject_id), y[sizeOf70PercentOfDataSet:])
    

def plotConfusionMatrix(confusion_matrix):
    print(confusion_matrix)
    df_cm = pd.DataFrame(confusion_matrix, range(3), range(3))
    #plt.figure(figsize = (10,7))
    sn.set(font_scale=1.4)#for label size
    sn.heatmap(df_cm, annot=True, cmap='Blues', annot_kws={"size": 16}, fmt='d')# font size
    plt.show()


loadSubjects(52)