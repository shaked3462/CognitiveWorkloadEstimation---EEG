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
from utils import loadSubjects, plot
from braindecode.datautil.signal_target import SignalAndTarget
import seaborn as sn
import pandas as pd

batch_size = 32
epoches = 400
model_type = 'shallow'
train_type = 'trialwise'
cuda = True
print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
print('\t\t{} {}'.format(model_type, train_type))
print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
# Enable logging
import logging
import importlib
importlib.reload(logging) # see https://stackoverflow.com/a/21475297/1469195
log = logging.getLogger()
log.setLevel('INFO')
import sys

logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                     level=logging.INFO, stream=sys.stdout)

singleTrainData = np.load("NirDataset\\cross_data_train.npy")
singleTestData = np.load("NirDataset\\cross_data_test.npy")
singleTrainLabels = np.load("NirDataset\\cross_labels_train.npy")
singleTestLabels = np.load("NirDataset\\cross_labels.npy")

trainingSampleSize = int(len(singleTrainData)*0.8)
validationSampleSize = int(len(singleTrainData)*0.2)
testSampleSize = int(len(singleTestData))
print("INFO : Training sample size: {}".format(trainingSampleSize))
print("INFO : Validation sample size: {}".format(validationSampleSize))
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

print(model.epochs_df)
np.save("DataForRestoration\\CrossSubject\\{}-{}-{}epoches".format(model_type, train_type, epoches), model.epochs_df.iloc[:])

# Evaluation
test_set = SignalAndTarget(singleTestData, y=singleTestLabels)

eval = model.evaluate(test_set.X, test_set.y)
print(eval)
print(eval['misclass'])
th.save(model, "crossModels\\{}-{}-cross-{}epoches-torch-model".format(model_type, train_type, epoches))

np.save("DataForRestoration\\CrossSubject\\{}-{}-{}epoches-testSetMisclass".format(model_type, train_type, epoches), eval['misclass'])

from sklearn.metrics import confusion_matrix

y_pred = model.predict_classes(test_set.X)
print("prediction {}".format(y_pred))
print("real labels {}".format(test_set.y))
confusion_matrix = confusion_matrix(test_set.y, y_pred)
print(confusion_matrix)

plot('accuracy', 'Plots\\CrossSubject\\', model, 0, model_type, train_type, epoches)
plot('confusionMatrix', 'Plots\\CrossSubject\\', model, confusion_matrix, model_type, train_type, epoches)