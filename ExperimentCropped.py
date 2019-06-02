import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

def loadSubjects(subjectNum):
    label0Counter = 0
    label1Counter = 0
    label2Counter = 0

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
            # print("X0 shape {}".format(X0.shape))
            # print("X1 shape {}".format(X1.shape))
            # print("X2 shape {}".format(X2.shape))

    X0, y0 = shuffle(X0, y0)
    X1, y1 = shuffle(X1, y1)
    X2, y2 = shuffle(X2, y2)
    examplesPerAllSubjects = np.amin([len(X0), len(X1), len(X2)])
    print("num of examples for each label {}".format(examplesPerAllSubjects))
    X = np.concatenate((X0[:examplesPerAllSubjects], X1[:examplesPerAllSubjects], X2[:examplesPerAllSubjects]), axis=0)
    y = np.concatenate((y0[:examplesPerAllSubjects], y1[:examplesPerAllSubjects], y2[:examplesPerAllSubjects]), axis=0)

    print("trying to run with 2.5 sec trials")

    return X, y
def plotConfusionMatrix(confusion_matrix):
    print(confusion_matrix)
    df_cm = pd.DataFrame(confusion_matrix, range(3), range(3))
    #plt.figure(figsize = (10,7))
    sn.set(font_scale=1.4)#for label size
    sn.heatmap(df_cm, annot=True, cmap='Blues', annot_kws={"size": 16}, fmt='d')# font size
    plt.show()

import logging
import os.path
import time
from collections import OrderedDict
import sys

import torch.nn.functional as F
from torch import optim

from braindecode.models.deep4 import Deep4Net
from braindecode.models.util import to_dense_prediction_model
from braindecode.datasets.bcic_iv_2a import BCICompetition4Set2A
from braindecode.experiments.experiment import Experiment
from braindecode.experiments.monitors import LossMonitor, MisclassMonitor, \
    RuntimeMonitor, CroppedTrialMisclassMonitor
from braindecode.experiments.stopcriteria import MaxEpochs, NoDecrease, Or
from braindecode.datautil.iterators import CropsFromTrialsIterator
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.datautil.splitters import split_into_two_sets
from braindecode.torch_ext.constraints import MaxNormDefaultConstraint
from braindecode.torch_ext.util import set_random_seeds, np_to_var
from braindecode.mne_ext.signalproc import mne_apply
from braindecode.datautil.signalproc import (bandpass_cnt,
                                             exponential_running_standardize)
from braindecode.datautil.trial_segment import create_signal_target_from_raw_mne
from braindecode.datautil.signal_target import SignalAndTarget

log = logging.getLogger(__name__)


def run_exp(epoches, batch_size, subject_num, model, cuda):
    # ival = [-500, 4000]
    max_increase_epochs = 160

    # Preprocessing
    X, y = loadSubjects(subject_num)
    X = X.astype(np.float32)
    y = y.astype(np.int64)
    X, y = shuffle(X, y)

    print("y")
    print(y)
    trainingSampleSize = int(len(X)*0.6)
    valudationSampleSize = int(len(X)*0.2)
    testSampleSize = int(len(X)*0.2)
    print("INFO : Training sample size: {}".format(trainingSampleSize))
    print("INFO : Validation sample size: {}".format(valudationSampleSize))
    print("INFO : Test sample size: {}".format(testSampleSize))


    train_set = SignalAndTarget(X[:trainingSampleSize], y=y[:trainingSampleSize])
    valid_set = SignalAndTarget(X[trainingSampleSize:(trainingSampleSize + valudationSampleSize)], y=y[trainingSampleSize:(trainingSampleSize + valudationSampleSize)])
    test_set = SignalAndTarget(X[(trainingSampleSize + valudationSampleSize):], y=y[(trainingSampleSize + valudationSampleSize):])

    set_random_seeds(seed=20190706, cuda=cuda)

    n_classes = 3
    n_chans = int(train_set.X.shape[1])
    
    if model == 'shallow':
        model = ShallowFBCSPNet(n_chans, n_classes, input_time_length=input_time_length,
                            final_conv_length=30).create_network()
    elif model == 'deep':
        model = Deep4Net(n_chans, n_classes, input_time_length=input_time_length,
                            final_conv_length=1).create_network()
    
    to_dense_prediction_model(model)
    if cuda:
        model.cuda()
    log.info("Model: \n{:s}".format(str(model)))
    dummy_input = np_to_var(train_set.X[:1, :, :, None])
    out = model(dummy_input)
    n_preds_per_input = out.cpu().data.numpy().shape[2]

    optimizer = optim.Adam(model.parameters())

    iterator = CropsFromTrialsIterator(batch_size=batch_size,
                                       input_time_length=input_time_length,
                                       n_preds_per_input=n_preds_per_input)

    stop_criterion = Or([MaxEpochs(max_epochs),
                         NoDecrease('valid_misclass', max_increase_epochs)])

    monitors = [LossMonitor(), MisclassMonitor(col_suffix='sample_misclass'),
                CroppedTrialMisclassMonitor(
                    input_time_length=input_time_length), RuntimeMonitor()]

    model_constraint = MaxNormDefaultConstraint()

    loss_function = lambda preds, targets: F.nll_loss(th.mean(preds, dim=2, keepdim=False), targets)

    exp = Experiment(model, train_set, valid_set, test_set, iterator=iterator,
                     loss_function=loss_function, optimizer=optimizer,
                     model_constraint=model_constraint,
                     monitors=monitors,
                     stop_criterion=stop_criterion,
                     remember_best_column='valid_misclass',
                     run_after_early_stop=True, cuda=cuda)
    exp.run()
    return exp

if __name__ == '__main__':
    np.set_printoptions(threshold=sys.maxsize, suppress=True)
    logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                            level=logging.DEBUG, stream=sys.stdout)
    model = 'shallow' #'shallow' or 'deep'
    max_epochs = 100
    batch_size = 8
    subject_num = 10
    print("INFO : {} Model, {} Epoches, {} Batch Size, {} Subjects".format(model, max_epochs, batch_size, subject_num))
    cuda = False
    exp = run_exp(max_epochs, batch_size, subject_num, model, cuda)
    log.info("Last 10 epochs")
    log.info("\n" + str(exp.epochs_df.iloc[-10:]))
    log.info("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    print(exp.epochs_df.iloc[:].shape)
    # np.save("{}_model-{}_epoches-{}_batch-{}_subjects".format(model, max_epochs, batch_size, subject_num), exp.epochs_df.iloc[:])
        
    try:
        plt.plot(exp.epochs_df.iloc[:,3], 'g.-', label='Train misclass')
        plt.plot(exp.epochs_df.iloc[:,4], 'b.-', label='Validation misclass')
        plt.plot(exp.epochs_df.iloc[:,5], 'r.-', label='Test misclass')
        plt.title('misclass rate / epoches')
        plt.xlabel('Epoches')
        plt.ylabel('Misclass')
        plt.legend(loc='best')
        plt.show()
    except:
        print("plot failed")