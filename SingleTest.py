import numpy as np
from sklearn.utils import shuffle
import sys
import torch as th
from torch import optim, nn
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from utils import loadSubjects, plotConfusionMatrix
from braindecode.datautil.signal_target import SignalAndTarget
import seaborn as sn
import pandas as pd

finetuning = False
batch_size = 32
epoches = 200
model_type = 'deep'
train_type = 'cropped'
cuda = True

subject_id = sys.argv[1]

# Enable logging
import logging
import importlib
importlib.reload(logging) # see https://stackoverflow.com/a/21475297/1469195
log = logging.getLogger()
log.setLevel('INFO')
import sys

logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                     level=logging.INFO, stream=sys.stdout)

singleTrainData = np.load("NirDataset\\Single\\subject{}_data_train.npy".format(subject_id))
singleTestData = np.load("NirDataset\\Single\\subject{}_data_test.npy".format(subject_id))
singleTrainLabels = np.load("NirDataset\\Single\\subject{}_labels_train.npy".format(subject_id))
singleTestLabels = np.load("NirDataset\\Single\\subject{}_labels.npy".format(subject_id))

trainingSampleSize = int(len(singleTrainData)*0.8)
valudationSampleSize = int(len(singleTrainData)*0.2)
testSampleSize = int(len(singleTestData))
print("INFO : Training sample size: {}".format(trainingSampleSize))
print("INFO : Validation sample size: {}".format(valudationSampleSize))
print("INFO : Test sample size: {}".format(testSampleSize))

train_set = SignalAndTarget(singleTrainData[:trainingSampleSize], y=singleTrainLabels[:trainingSampleSize])
valid_set = SignalAndTarget(singleTrainData[trainingSampleSize:], y=singleTrainLabels[trainingSampleSize:])

# Create the model
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.models.deep4 import Deep4Net
from braindecode.torch_ext.util import set_random_seeds

set_random_seeds(seed=20170629, cuda=cuda)
n_classes = 3
in_chans = train_set.X.shape[1]
print("INFO : in_chans: {}".format(in_chans))

# final_conv_length = auto ensures we only get a single output in the time dimension
if train_type == 'trialwise':
    input_time_length = train_set.X.shape[2]
    if model_type == 'shallow':
        model = ShallowFBCSPNet(in_chans=in_chans, n_classes=n_classes,
                            input_time_length=input_time_length,
                            final_conv_length='auto')
    else:
        model = Deep4Net(in_chans=in_chans, n_classes=n_classes,
                            input_time_length=input_time_length,
                            final_conv_length='auto')
else: # cropped
    if model_type == 'shallow':
        model = ShallowFBCSPNet(in_chans=in_chans, n_classes=n_classes,
                            input_time_length=None,
                            final_conv_length=1)
    else:
        model = Deep4Net(in_chans=in_chans, n_classes=n_classes,
                            input_time_length=None,
                            final_conv_length=1)
if cuda:
    model.cuda()

from braindecode.torch_ext.optimizers import AdamW
import torch.nn.functional as F
if model_type == 'shallow':
    optimizer = AdamW(model.parameters(), lr=0.0625 * 0.01, weight_decay=0)
else:
    optimizer = AdamW(model.parameters(), lr=1*0.01, weight_decay=0.5*0.001) # these are good values for the deep model

if finetuning == True:
    path_to_classifier = "torchModelsCrossSubjects\{}-{}-52subjects-2.5sec-800epoches-torch_model".format(model_type, train_type)
    if th.cuda.is_available():
        print('Cuda is available.')
        checkpoint = th.load(path_to_classifier).state_dict()
    else:
        print('Cuda is not available.')
        checkpoint = th.load(path_to_classifier, map_location='cpu').state_dict()
    np.set_printoptions(suppress=True, threshold=np.inf)
    model.network.load_state_dict(checkpoint)

if train_type == 'trialwise' :
    model.compile(loss=F.nll_loss, optimizer=optimizer, iterator_seed=1)
else: # cropped 
    model.compile(loss=F.nll_loss, optimizer=optimizer, iterator_seed=1, cropped=True)

# Compile model exactly the same way as when you trained it

print("INFO : Epochs: {}".format(epoches))
print("INFO : Batch Size: {}".format(batch_size))


# Fit model exactly the same way as when you trained it (omit any optional params though)
if train_type == 'trialwise':
    print(model.fit(train_set.X, train_set.y, epochs=epoches, batch_size=batch_size, scheduler='cosine',
            validation_data=(valid_set.X, valid_set.y),))
else: # cropped 
    input_time_length = 450
    print(model.fit(train_set.X, train_set.y, epochs=epoches, batch_size=batch_size, scheduler='cosine',
            input_time_length=input_time_length,
            validation_data=(valid_set.X, valid_set.y),))
if finetuning == True:
    print('Loaded saved torch model from "{}".'.format(path_to_classifier))

print(model.epochs_df)
# np.save("finetuneCrossSubjects\{}-{}-singleSubjectNum{}-2.5sec-{}epoches".format(model_type, train_type, single_subject_num, epoches), model.epochs_df.iloc[:])

# Evaluation
test_set = SignalAndTarget(singleTestData, y=singleTestLabels)

eval = model.evaluate(test_set.X, test_set.y)
print(eval)
print(eval['misclass'])
# np.save("finetuneCrossSubjects\{}-{}-singleSubjectNum{}-2.5sec-{}epoches-testSetMisclass".format(model_type, train_type, single_subject_num, epoches), eval['misclass'])

from sklearn.metrics import confusion_matrix

try:
    print("prediction")
    y_pred = model.predict_classes(test_set.X)
    print(y_pred)
    print("real labels")
    print(test_set.y)
    confusion_matrix = confusion_matrix(test_set.y, y_pred)
    print(confusion_matrix)
except:
    print("predict_classes method failed")

# plotMisclass()

def plotMisclass():
    plt.plot(model.epochs_df.iloc[:,2], 'g-', label='Train misclass')
    plt.plot(model.epochs_df.iloc[:,3], 'b-', label='Validation misclass')
    model# plt.plot(exp.epochs_df.iloc[:,5], 'r.-', label='Test misclass')
    plt.title('misclass rate / epoches')
    plt.xlabel('Epoches')
    plt.ylabel('Misclass')
    plt.legend(loc='best')
    plt.show()
    # plt.savefig("finetuneCrossSubjects\{}-{}-singleSubjectNum{}-2.5sec-{}epoches-plot-misclass.png".format(model_type, train_type, single_subject_num, epoches), bbox_inches='tight')
    plt.close()

def plotAccuracy():
    plt.plot(1-model.epochs_df.iloc[:,2], 'g-', label='Train accuracy')
    plt.plot(1-model.epochs_df.iloc[:,3], 'b-', label='Validation accuracy')
    model# plt.plot(exp.epochs_df.iloc[:,5], 'r.-', label='Test misclass')
    plt.title('accuracy rate / epoches')
    plt.xlabel('Epoches')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.show()
    # plt.savefig("finetuneCrossSubjects\{}-{}-singleSubjectNum{}-2.5sec-{}epoches-plot-accuracy.png".format(model_type, train_type, single_subject_num, epoches), bbox_inches='tight')
    plt.close()

def plotConfusionMatrixFinetune(confusion_matrix):
    array = confusion_matrix       
    df_cm = pd.DataFrame(array, range(3),
                    range(3))
    #plt.figure(figsize = (10,7))
    sn.set(font_scale=1.4)#for label size
    sn.heatmap(df_cm, annot=True, cmap='Blues', annot_kws={"size": 16}, fmt='d')# font size
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    plt.show()
    # plt.savefig("finetuneCrossSubjects\{}-{}-singleSubjectNum{}-2.5sec-{}epoches-confusion_matrix.png".format(model_type, train_type, single_subject_num, epoches), bbox_inches='tight')
    plt.close()

plotAccuracy()
plotConfusionMatrixFinetune(confusion_matrix)