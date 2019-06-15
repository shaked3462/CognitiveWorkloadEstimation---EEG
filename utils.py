import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from sklearn.metrics import confusion_matrix

def loadSubjects(subjectNum, single_subject, single_subject_num):
    label0Counter = 0
    label1Counter = 0
    label2Counter = 0

    if single_subject == True:
        print("loading data for subject {}".format(single_subject_num))
        X = np.load("NirDataset\\croppedDataForSubject{}.npy".format(single_subject_num))
        y = np.load("NirDataset\\croppedLabelsForSubject{}.npy".format(single_subject_num))
    else:        
        print("loading data for subject 1")
        X = np.load("NirDataset\\croppedDataForSubject1.npy")
        y = np.load("NirDataset\\croppedLabelsForSubject1.npy")

    for i in range(0,len(y)):
        if y[i] == 0:
            label0Counter += 1
        if y[i] == 1:
            label1Counter += 1
        if y[i] == 2:
            label2Counter += 1
    print("INFO : label 0 examples: {}, label 1 examples: {}, label 2 examples: {}".format(label0Counter, label1Counter, label2Counter))
    examplesPerSubjecti = np.amin([label0Counter, label1Counter, label2Counter])
    print("INFO : used {} examples from each label".format(examplesPerSubjecti))
    print(X.shape)
    X0 = X[:label0Counter,:,:]
    X1 = X[label0Counter:label0Counter+label1Counter,:,:]
    X2 = X[label0Counter+label1Counter:,:,:]
    y0 = y[:label0Counter]
    y1 = y[label0Counter:label0Counter+label1Counter]
    y2 = y[label0Counter+label1Counter:]

    if subjectNum > 1:
        for i in range(2,subjectNum+1):
            if (i == 15) or (i == 24) or (i == 25) or (i == 34) or (i == 40): #subjects that were excluded from experiment.
                continue
            print("loading data for subject {}".format(i))
            Xtmp = np.load("NirDataset\\croppedDataForSubject{}.npy".format(i))
            ytmp = np.load("NirDataset\\croppedLabelsForSubject{}.npy".format(i))

            label0Counter = 0
            label1Counter = 0
            label2Counter = 0
            for i in range(0,len(ytmp)):
                if ytmp[i] == 0:
                    label0Counter += 1
                if ytmp[i] == 1:
                    label1Counter += 1
                if ytmp[i] == 2:
                    label2Counter += 1
            print("INFO : label 0 examples: {}, label 1 examples: {}, label 2 examples: {}".format(label0Counter, label1Counter, label2Counter))
            examplesPerSubjecti = np.amin([label0Counter, label1Counter, label2Counter])
            print("INFO : used {} examples from each label".format(examplesPerSubjecti))
    
            Xtmp0 = Xtmp[:label0Counter,:,:]
            Xtmp1 = Xtmp[label0Counter:label0Counter+label1Counter,:,:]
            Xtmp2 = Xtmp[label0Counter+label1Counter:,:,:]
            ytmp0 = ytmp[:label0Counter]
            ytmp1 = ytmp[label0Counter:label0Counter+label1Counter]
            ytmp2 = ytmp[label0Counter+label1Counter:]
            X0 = np.concatenate((X0, Xtmp0), axis=0)
            X1 = np.concatenate((X1, Xtmp1), axis=0)
            X2 = np.concatenate((X2, Xtmp2), axis=0)
            y0 = np.concatenate((y0, ytmp0), axis=0)
            y1 = np.concatenate((y1, ytmp1), axis=0)
            y2 = np.concatenate((y2, ytmp2), axis=0)

    X0, y0 = shuffle(X0, y0)
    X1, y1 = shuffle(X1, y1)
    X2, y2 = shuffle(X2, y2)
    examplesPerAllSubjects = np.amin([len(X0), len(X1), len(X2)])
    print("num of examples for each label {}".format(examplesPerAllSubjects))
    X = np.concatenate((X0[:examplesPerAllSubjects], X1[:examplesPerAllSubjects], X2[:examplesPerAllSubjects]), axis=0)
    y = np.concatenate((y0[:examplesPerAllSubjects], y1[:examplesPerAllSubjects], y2[:examplesPerAllSubjects]), axis=0)

    return X, y

def plotConfusionMatrix(confusion_matrix):
    print(confusion_matrix)
    df_cm = pd.DataFrame(confusion_matrix, range(3), range(3))
    #plt.figure(figsize = (10,7))
    sn.set(font_scale=1.4)#for label size
    sn.heatmap(df_cm, annot=True, cmap='Blues', annot_kws={"size": 16}, fmt='d')# font size
    plt.show()
