import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from utils import loadSubjects, plotConfusionMatrix

import logging
import os.path
import time
from collections import OrderedDict
import sys

import torch.nn.functional as F
from torch import optim
import torch as th
from braindecode.models.deep4 import Deep4Net
from braindecode.models.eegnet import EEGNetv4
from braindecode.models.eegnet import EEGNet
from braindecode.datasets.bcic_iv_2a import BCICompetition4Set2A
from braindecode.experiments.experiment import Experiment
from braindecode.experiments.monitors import LossMonitor, MisclassMonitor, \
    RuntimeMonitor
from braindecode.experiments.stopcriteria import MaxEpochs, NoDecrease, Or
from braindecode.datautil.iterators import BalancedBatchSizeIterator
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


def run_exp(epoches, batch_size, subject_num, model_type, cuda, single_subject, single_subject_num):
    # ival = [-500, 4000]
    max_increase_epochs = 160
     

    # Preprocessing
    X, y = loadSubjects(subject_num, single_subject, single_subject_num)
    X = X.astype(np.float32)
    y = y.astype(np.int64)
    X, y = shuffle(X, y)
    
    trial_length = X.shape[2]
    print("trial_length " + str(trial_length))
    print("trying to run with {} sec trials ".format((trial_length - 1) / 256))
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
    input_time_length = train_set.X.shape[2]
    if model_type == 'shallow':
        model = ShallowFBCSPNet(n_chans, n_classes, input_time_length=input_time_length,
                            final_conv_length='auto').create_network()
    elif model_type == 'deep':
        model = Deep4Net(n_chans, n_classes, input_time_length=input_time_length,
                            final_conv_length='auto').create_network()
    elif model_type == 'eegnet':
        model = EEGNetv4(n_chans, n_classes, input_time_length=input_time_length,
                            final_conv_length='auto').create_network()
    if cuda:
        model.cuda()
    log.info("Model: \n{:s}".format(str(model)))

    optimizer = optim.Adam(model.parameters())

    iterator = BalancedBatchSizeIterator(batch_size=batch_size)

    stop_criterion = Or([MaxEpochs(max_epochs),
                         NoDecrease('valid_misclass', max_increase_epochs)])

    monitors = [LossMonitor(), MisclassMonitor(), RuntimeMonitor()]

    model_constraint = MaxNormDefaultConstraint()

    exp = Experiment(model, train_set, valid_set, test_set, iterator=iterator,
                     loss_function=F.nll_loss, optimizer=optimizer,
                     model_constraint=model_constraint,
                     monitors=monitors,
                     stop_criterion=stop_criterion,
                     remember_best_column='valid_misclass',
                     run_after_early_stop=True, cuda=cuda)
    exp.run()
    # th.save(model, "models\{}-cropped-singleSubjectNum{}-{}sec-{}epoches-torch_model".format(model_type, single_subject_num, ((trial_length - 1) / 256), epoches))
    return exp

if __name__ == '__main__':
    np.set_printoptions(threshold=sys.maxsize, suppress=True)
    logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                            level=logging.DEBUG, stream=sys.stdout)
    model = 'shallow' #'shallow' or 'deep'
    max_epochs = 800
    batch_size = 16
    subject_num = 1
    single_subject = True
    single_subject_num = sys.argv[1]
    print("single subject num " + single_subject_num)
    trial_length = 2.5
    print("INFO : {} Model, {} Epoches, {} Batch Size, {} Subjects".format(model, max_epochs, batch_size, subject_num))
    if single_subject == True:
        print("INFO : Single subject num {}".format(single_subject_num))
    cuda = True
    exp = run_exp(max_epochs, batch_size, subject_num, model, cuda, single_subject, single_subject_num)
    log.info("epochs")
    log.info("\n" + str(exp.epochs_df.iloc[:]))
    log.info("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    print(exp.epochs_df.iloc[:].shape)
    np.save("single_subjects\{}-cropped-singleSubjectNum{}-2.5sec-{}epoches".format(model, single_subject_num, max_epochs), exp.epochs_df.iloc[:])
        
    # try:
    plt.plot(exp.epochs_df.iloc[:,3], 'g.-', label='Train misclass')
    plt.plot(exp.epochs_df.iloc[:,4], 'b.-', label='Validation misclass')
    plt.plot(exp.epochs_df.iloc[:,5], 'r.-', label='Test misclass')
    plt.title('misclass rate / epoches')
    plt.xlabel('Epoches')
    plt.ylabel('Misclass')
    plt.legend(loc='best')
    # plt.show()
    if single_subject == True:
        plt.savefig("single_subjects\{}-cropped-subject{}-{}sec-{}epoches.png".format(model, single_subject_num, trial_length, max_epochs), bbox_inches='tight')
        #plt.savefig("single_subjects\{}-cropped-subject{}-{}sec.pdf".format(model, single_subject_num, trial_length), bbox_inches='tight')
    else:    
        plt.savefig("single_subjects\{}-cropped-{}subjects-{}sec-{}epoches.png".format(model, subject_num, trial_length, max_epochs), bbox_inches='tight')
        #plt.savefig("single_subjects\{}-cropped-{}subjects-{}sec.pdf".format(model, subject_num, trial_length), bbox_inches='tight')
    # except:
    #     print("plot failed")