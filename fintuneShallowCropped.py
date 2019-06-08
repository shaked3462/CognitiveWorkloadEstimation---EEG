import numpy as np
from sklearn.utils import shuffle
import sys
import torch as th
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy


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

    return X, y
def plotConfusionMatrix(confusion_matrix):
    print(confusion_matrix)
    df_cm = pd.DataFrame(confusion_matrix, range(3), range(3))
    #plt.figure(figsize = (10,7))
    sn.set(font_scale=1.4)#for label size
    sn.heatmap(df_cm, annot=True, cmap='Blues', annot_kws={"size": 16}, fmt='d')# font size
    plt.show()

single_subject_num = sys.argv[1]
X, y = loadSubjects(1, True, single_subject_num)
batch_size = 64
epoches = 800
model_type = 'shallow'
train_type = 'cropped'

# Enable logging
import logging
import importlib
importlib.reload(logging) # see https://stackoverflow.com/a/21475297/1469195
log = logging.getLogger()
log.setLevel('INFO')
import sys

logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                     level=logging.INFO, stream=sys.stdout)

X = X.astype(np.float32)
y = y.astype(np.int64)

X, y = shuffle(X, y)
print("y")
print(y)
from braindecode.datautil.signal_target import SignalAndTarget
trainingSampleSize = int(len(X)*0.7)
valudationSampleSize = int(len(X)*0.1)
testSampleSize = int(len(X)*0.2)
print("INFO : Training sample size: {}".format(trainingSampleSize))
print("INFO : Validation sample size: {}".format(valudationSampleSize))
print("INFO : Test sample size: {}".format(testSampleSize))


train_set = SignalAndTarget(X[:trainingSampleSize], y=y[:trainingSampleSize])
valid_set = SignalAndTarget(X[trainingSampleSize:(trainingSampleSize + valudationSampleSize)], y=y[trainingSampleSize:(trainingSampleSize + valudationSampleSize)])

# Create the model
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.models.deep4 import Deep4Net
from torch import nn
from braindecode.torch_ext.util import set_random_seeds

# Set if you want to use GPU
# You can also use torch.cuda.is_available() to determine if cuda is available on your machine.
cuda = True
set_random_seeds(seed=20170629, cuda=cuda)
n_classes = 3
in_chans = train_set.X.shape[1]
print("INFO : in_chans: {}".format(in_chans))
print("INFO : input_time_length: {}".format(train_set.X.shape[2]))

# final_conv_length = auto ensures we only get a single output in the time dimension
if model_type == 'shallow':
    model = ShallowFBCSPNet(in_chans=in_chans, n_classes=n_classes,
                        input_time_length=train_set.X.shape[2],
                        final_conv_length='auto')
else:
    model = Deep4Net(in_chans=in_chans, n_classes=n_classes,
                        input_time_length=train_set.X.shape[2],
                        final_conv_length='auto')
path_to_classifier = "models\{}-{}-52subjects-2.5sec-800epoches-torch_model".format(model_type, train_type)


if cuda:
    model.cuda()



from braindecode.torch_ext.optimizers import AdamW
import torch.nn.functional as F
if model_type == 'shallow':
    optimizer = AdamW(model.parameters(), lr=0.0625 * 0.01, weight_decay=0)
else:
    optimizer = AdamW(model.parameters(), lr=1*0.01, weight_decay=0.5*0.001) # these are good values for the deep model

if torch.cuda.is_available():
    print('Cuda is available.')
    checkpoint = torch.load(path_to_classifier).state_dict()
else:
    print('Cuda is not available.')
    checkpoint = torch.load(path_to_classifier, map_location='cpu')
np.set_printoptions(suppress=True, threshold=np.inf)

# print("checkpoint")
# print(checkpoint)


# print("checkpoint.state_dict()")
# print(checkpoint.state_dict())

model.network.load_state_dict(checkpoint)
# optimizer.load_state_dict(checkpoint.optimizer.state_dict())

# Compile model exactly the same way as when you trained it
model.compile(loss=F.nll_loss, optimizer=optimizer, iterator_seed=1)

print("INFO : Epochs: {}".format(epoches))
print("INFO : Batch Size: {}".format(batch_size))


# Fit model exactly the same way as when you trained it (omit any optional params though)
print(model.fit(train_set.X, train_set.y, epochs=epoches, batch_size=batch_size, scheduler='cosine',
         validation_data=(valid_set.X, valid_set.y),))

print('Loaded saved torch model from "{}".'.format(path_to_classifier))

print(model.epochs_df)
np.save("finetuning\{}-{}-singleSubjectNum{}-2.5sec-{}epoches".format(model_type, train_type, single_subject_num, epoches), model.epochs_df.iloc[:])

# Evaluation
test_set = SignalAndTarget(X[(trainingSampleSize + valudationSampleSize):], y=y[(trainingSampleSize + valudationSampleSize):])

eval = model.evaluate(test_set.X, test_set.y)
print(eval)
print(eval['misclass'])
np.save("finetuning\{}-{}-singleSubjectNum{}-2.5sec-{}epoches-testSetMisclass".format(model_type, train_type, single_subject_num, epoches), eval['misclass'])

from sklearn.metrics import confusion_matrix

try:
    print("prediction")
    y_pred = model.predict(test_set.X)
    print(y_pred)
    print("real labels")
    print(test_set.y)
    confusion_matrix = confusion_matrix(test_set.y, y_pred)
    print(confusion_matrix)
except:
    try:
        y_pred = model.predict_classes(test_set.X)
        print(y_pred)
        print("real labels")
        print(test_set.y)
        confusion_matrix = confusion_matrix(test_set.y, y_pred)
        print(confusion_matrix)
    except:
        print("predict_classes method failed")


import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

array = confusion_matrix       
df_cm = pd.DataFrame(array, range(3),
                  range(3))
#plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, annot=True, cmap='Blues', annot_kws={"size": 16}, fmt='d')# font size
pd.set_option('display.float_format', lambda x: '%.3f' % x)
# plt.show()
plt.savefig("finetuning\{}-{}-singleSubjectNum{}-2.5sec-{}epoches-confusion_matrix.png".format(model_type, train_type, single_subject_num, epoches), bbox_inches='tight')
plt.close()

plt.plot(model.epochs_df.iloc[:,2], 'g-', label='Train misclass')
plt.plot(model.epochs_df.iloc[:,3], 'b-', label='Validation misclass')
model# plt.plot(exp.epochs_df.iloc[:,5], 'r.-', label='Test misclass')
plt.title('misclass rate / epoches')
plt.xlabel('Epoches')
plt.ylabel('Misclass')
plt.legend(loc='best')
# plt.show()
plt.savefig("finetuning\{}-{}-singleSubjectNum{}-2.5sec-{}epoches-plot-misclass.png".format(model_type, train_type, single_subject_num, epoches), bbox_inches='tight')
plt.close()

plt.plot(1-model.epochs_df.iloc[:,2], 'g-', label='Train accuracy')
plt.plot(1-model.epochs_df.iloc[:,3], 'b-', label='Validation accuracy')
model# plt.plot(exp.epochs_df.iloc[:,5], 'r.-', label='Test misclass')
plt.title('accuracy rate / epoches')
plt.xlabel('Epoches')
plt.ylabel('Accuracy')
plt.legend(loc='best')
# plt.show()
plt.savefig("finetuning\{}-{}-singleSubjectNum{}-2.5sec-{}epoches-plot-accuracy.png".format(model_type, train_type, single_subject_num, epoches), bbox_inches='tight')
plt.close()
