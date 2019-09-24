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
        X = X.astype(np.float32)
        y = y.astype(np.int64)

        print(X.shape)
        sizeOf70PercentOfTrainDataSet = int(X.shape[0]*7/10)

        np.save("NirDataset\\Single\\subject{}_data_train".format(subject_id), X[:sizeOf70PercentOfTrainDataSet])
        np.save("NirDataset\\Single\\subject{}_data_test".format(subject_id), X[sizeOf70PercentOfTrainDataSet:])
        np.save("NirDataset\\Single\\subject{}_labels_train".format(subject_id), y[:sizeOf70PercentOfTrainDataSet])
        np.save("NirDataset\\Single\\subject{}_labels".format(subject_id), y[sizeOf70PercentOfTrainDataSet:])
    

def procDataForCross():
    for subject_id in range(1,52+1):
        if (subject_id == 15) or (subject_id == 24) or (subject_id == 25) or (subject_id == 34) or (subject_id == 40): #subjects that were excluded from experiment.
            continue
        Xtmp = np.load("NirDataset\\Single\\subject{}_data_train.npy".format(subject_id))
        ytmp = np.load("NirDataset\\Single\\subject{}_labels_train.npy".format(subject_id))

        print("subject_id: {}".format(subject_id))
        print("Xtmp shape: {}".format(Xtmp.shape))
        print("ytmp shape: {}".format(ytmp.shape))
        print("ytmp: {}".format(ytmp))
        if subject_id == 1:
            X = Xtmp
            y = ytmp
        else:
            X = np.concatenate((X, Xtmp), axis=0)
            y = np.concatenate((y, ytmp), axis=0)
        
        print("y shape: {}".format(y.shape))
        print("y: {}".format(y))
    
    X, y = shuffle(X, y)
    X = X.astype(np.float32)
    y = y.astype(np.int64)

    sizeOf70PercentOfTrainDataSet = int(X.shape[0]*7/10)
    print("sizeOf70PercentOfTrainDataSet: {}".format(sizeOf70PercentOfTrainDataSet))

    np.save("NirDataset\\cross_data_train", X[:sizeOf70PercentOfTrainDataSet])
    np.save("NirDataset\\cross_data_test", X[sizeOf70PercentOfTrainDataSet:])
    np.save("NirDataset\\cross_labels_train", y[:sizeOf70PercentOfTrainDataSet])
    np.save("NirDataset\\cross_labels", y[sizeOf70PercentOfTrainDataSet:])


# def plotConfusionMatrix(confusion_matrix):
#     print(confusion_matrix)
#     df_cm = pd.DataFrame(confusion_matrix, range(3), range(3))
#     #plt.figure(figsize = (10,7))
#     sn.set(font_scale=1.4)#for label size
#     sn.heatmap(df_cm, annot=True, cmap='Blues', annot_kws={"size": 16}, fmt='d')# font size
#     plt.show()

def plot(plotType, dirPath, model, confusion_matrix, model_type, train_type, epoches): # dir path of format /dir1/dir2/
    if plotType == 'confusionMatrix':
        array = confusion_matrix       
        df_cm = pd.DataFrame(array, range(3),
                                    range(3))
        #plt.figure(figsize = (10,7))
        sn.set(font_scale=1.4)#for label size
        sn.heatmap(df_cm, annot=True, cmap='Blues', annot_kws={"size": 16}, fmt='d')# font size
        pd.set_option('display.float_format', lambda x: '%.3f' % x)
        # plt.show()
        plt.savefig("{}{}-{}-{}epoches-confusion_matrix.png".format(dirPath, model_type, train_type, epoches), bbox_inches='tight')
        plt.close()
    else:    
        if plotType == 'accuracy':
            plt.plot(1-model.epochs_df.iloc[:,2], 'g-', label='Train accuracy')
            plt.plot(1-model.epochs_df.iloc[:,3], 'b-', label='Validation accuracy')
            plt.title('accuracy rate / epoches')
            plt.ylabel('Accuracy')
        else: #misclass
            plt.plot(model.epochs_df.iloc[:,2], 'g-', label='Train misclass')
            plt.plot(model.epochs_df.iloc[:,3], 'b-', label='Validation misclass')
            plt.title('misclass rate / epoches')
            plt.ylabel('Misclass')
        plt.xlabel('Epoches')
        plt.legend(loc='best')
        # plt.show()
        plt.savefig("{}{}-{}-{}epoches-accuracy.png".format(dirPath,model_type, train_type, epoches), bbox_inches='tight')
        plt.close()