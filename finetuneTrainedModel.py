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



single_subject_num = sys.argv[1]
X, y = loadSubjects(1, True, single_subject_num)
batch_size = 64
epoches = 800
model_type = 'shallow'
train_type = 'trialwise'

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
path_to_classifier = "torchModelsCrossSubjects\{}-{}-52subjects-2.5sec-800epoches-torch_model".format(model_type, train_type)


if cuda:
    model.cuda()



from braindecode.torch_ext.optimizers import AdamW
import torch.nn.functional as F
if model_type == 'shallow':
    optimizer = AdamW(model.parameters(), lr=0.0625 * 0.01, weight_decay=0)
else:
    optimizer = AdamW(model.parameters(), lr=1*0.01, weight_decay=0.5*0.001) # these are good values for the deep model

if th.cuda.is_available():
    print('Cuda is available.')
    checkpoint = th.load(path_to_classifier).state_dict()
else:
    print('Cuda is not available.')
    checkpoint = th.load(path_to_classifier, map_location='cpu')
np.set_printoptions(suppress=True, threshold=np.inf)

model.network.load_state_dict(checkpoint)

# Compile model exactly the same way as when you trained it
model.compile(loss=F.nll_loss, optimizer=optimizer, iterator_seed=1)

print("INFO : Epochs: {}".format(epoches))
print("INFO : Batch Size: {}".format(batch_size))


# Fit model exactly the same way as when you trained it (omit any optional params though)
print(model.fit(train_set.X, train_set.y, epochs=epoches, batch_size=batch_size, scheduler='cosine',
         validation_data=(valid_set.X, valid_set.y),))

print('Loaded saved torch model from "{}".'.format(path_to_classifier))

print(model.epochs_df)
# np.save("finetuneCrossSubjects\{}-{}-singleSubjectNum{}-2.5sec-{}epoches".format(model_type, train_type, single_subject_num, epoches), model.epochs_df.iloc[:])

# Evaluation
test_set = SignalAndTarget(X[(trainingSampleSize + valudationSampleSize):], y=y[(trainingSampleSize + valudationSampleSize):])

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

plotMisclass()
plotAccuracy()
plotConfusionMatrixFinetune(confusion_matrix)

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
