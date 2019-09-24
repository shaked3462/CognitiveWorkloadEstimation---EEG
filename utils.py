import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
from matplotlib import colors as mcolors
from sklearn.metrics import confusion_matrix

def loadEegDataFromMatlabtoNumpy():
    subject_num = 52
    cropped_trial_length = 640 # 2.5 sec at 256Hz sampling rate

    for subject_id in range(1,subject_num + 1):
        if (subject_id == 15) or (subject_id == 24) or (subject_id == 25) or (subject_id == 34) or (subject_id == 40): #subjects that were excluded from experiment.
            continue
        # try:
        print("XXXXXXXXXX STARTING DATA PROCESSING FOR SUBJECT {} XXXXXXXXXX".format(subject_id))
        if (subject_id < 10):
            mat_contents = sio.loadmat("NirDataset\subject{}\Subject_00{}_Main_Exp_Segmented_shaked.mat".format(subject_id, subject_id))
            answer_vec_content = sio.loadmat("NirDataset\subject{}\\answer_vec.mat".format(subject_id))
            respMat_content = sio.loadmat("NirDataset\subject{}\\respMat_Subject_00{}".format(subject_id, subject_id))
        else:
            mat_contents = sio.loadmat("NirDataset\subject{}\Subject_0{}_Main_Exp_Segmented_shaked.mat".format(subject_id, subject_id))
            answer_vec_content = sio.loadmat("NirDataset\subject{}\\answer_vec.mat".format(subject_id))
            respMat_content = sio.loadmat("NirDataset\subject{}\\respMat_Subject_0{}".format(subject_id, subject_id))

        respMat = respMat_content.get('respMat')
        if subject_id > 44:
            answer_vec = answer_vec_content.get('curr_ans_vec')
        else:
            answer_vec = answer_vec_content.get('answer_vec')

        segments = mat_contents.get('segments')

        np.set_printoptions(suppress=True)
        first = 0
        # data = np.zeros((1,62,cropped_trial_length + 1)) # XXXXXXXXXXXXXXXXX NEED TO ADD DURATION OF TRIAL AS ANOTHER PARAM
        # labels = np.zeros((1))
        for i in range(0,36): # question number
            # try:
            if answer_vec[i] == 0: #wrong answer
                print("Q{}: skipping due to wrong answer".format(i))
                continue 
            timeToAnswerQuestion = respMat[i][3][0][0]
            currQuestionDuration = len(segments[i][0][0]) # number of timestamps in original recording
            # print("curr questions duration {}".format(currQuestionDuration))
            last_crop_start = 0
            numOfCropsForCurrQuestion = int(currQuestionDuration/cropped_trial_length)
            for b in range(0,numOfCropsForCurrQuestion):
                currCrop = np.zeros((1,62,cropped_trial_length + 1))
                for k in range(0,cropped_trial_length):
                    # if last_crop_start + cropped_trial_length >= currQuestionDuration:
                    #     print("Q{}: last crop start {}, cropped trial length {}, curr question duration {}. breaking".format(i, last_crop_start, cropped_trial_length, currQuestionDuration))
                    #     break
                    # numOfCropsForCurrQuestion += 1
                    for j in range(0,62):
                        currCrop[0,j,k] = segments[i][0][j,(last_crop_start + k)]
                        currCrop[0,j,cropped_trial_length] = timeToAnswerQuestion
                last_crop_start += cropped_trial_length
                if first == 0:
                    data = currCrop
                    labels = np.array([int((i-1)/12)])
                    first += 1
                else:
                    data = np.concatenate((data, currCrop), axis=0)
                    labels = np.concatenate((labels, np.array([int((i-1)/12)])), axis=0)
            print("Q{}: number of crops {}".format(i, numOfCropsForCurrQuestion))
            # except:
            #     continue
            

        labels = labels.astype(np.int64)
        print(labels.shape)
        print(data.shape)
        np.save("NirDataset/croppedDataForSubject{}".format(subject_id), data[1:,:,:])
        np.save("NirDataset/croppedLabelsForSubject{}".format(subject_id), labels[1:])
        print("finished saving data for subject {}".format(subject_id))

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