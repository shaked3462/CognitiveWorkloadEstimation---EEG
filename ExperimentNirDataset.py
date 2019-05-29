import numpy as np
from sklearn.utils import shuffle

print("loading data for subject 1")
X = np.load("NirDataset\\croppedDataForSubject1.npy")
y = np.load("NirDataset\\croppedLabelsForSubject1.npy")
#for i in range(2,20):
#    if (i == 6) or (i == 14) or (i == 26):
#        continue
#    print("loading data for subject {}".format(i))
#    Xtmp = np.load("NirDataset\\croppedDataForSubject{}.npy".format(i))
#    ytmp = np.load("NirDataset\\croppedLabelsForSubject{}.npy".format(i))
#    X = np.concatenate((X, Xtmp), axis=0)
#    y = np.concatenate((y, ytmp), axis=0)
#    print(X.shape)
#    print(y.shape)
## print("INFO : SUBJECTS NUM : 41 (44 without 6,14,26)")
#print("INFO : SUBJECTS NUM : 18 (20 without 6,14)")

# print(Z.shape)
label0Counter = 0
label1Counter = 0
label2Counter = 0

for i in range(0,len(y)):
    if y[i] == 0:
        label0Counter += 1
    if y[i] == 1:
        label1Counter += 1
    if y[i] == 2:
        label2Counter += 1
print("INFO : label 0 examples: {}, label 1 examples: {}, label 2 examples: {}".format(label0Counter, label1Counter, label2Counter))
print("INFO : used {} examples from each label".format(label0Counter))

print(X.shape)
#X0 = X[:label0Counter,:,:]
#X1 = X[label0Counter:label1Counter,:,:]
#X2 = X[label0Counter+label1Counter:,:,:]
Xsplitted = np.split(X, [label0Counter, label1Counter, label2Counter])
ysplitted = np.split(y, [label0Counter, label1Counter, label2Counter])
X0 = Xsplitted[0]
X1 = Xsplitted[1]
X2 = Xsplitted[2]
y0 = ysplitted[0]
y1 = ysplitted[1]
y2 = ysplitted[2]



X1, y1 = shuffle(X1, y1)
X2, y2 = shuffle(X2, y2)

# Enable logging
import logging
import importlib
importlib.reload(logging) # see https://stackoverflow.com/a/21475297/1469195
log = logging.getLogger()
log.setLevel('INFO')
import sys

logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                     level=logging.INFO, stream=sys.stdout)


X = np.concatenate((X0, X1[:label0Counter], X2[:label0Counter]), axis=0)
y = np.concatenate((y0, y1[:label0Counter], y2[:label0Counter]), axis=0)

print("INFO : running with trials 2.5 sec long")
X = X[:,:,:640]

X = X.astype(np.float32)
y = y.astype(np.int64)


X, y = shuffle(X, y)

from braindecode.datautil.signal_target import SignalAndTarget
trainingSampleSize = int(len(X)*0.5)
valudationSampleSize = int(len(X)*0.2)
testSampleSize = int(len(X)*0.3)
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
cuda = False
set_random_seeds(seed=20170629, cuda=cuda)
n_classes = 3
in_chans = train_set.X.shape[1]
print("INFO : in_chans: {}".format(in_chans))
print("INFO : input_time_length: {}".format(train_set.X.shape[2]))

# final_conv_length = auto ensures we only get a single output in the time dimension
model = Deep4Net(in_chans=in_chans, n_classes=n_classes,
                        input_time_length=train_set.X.shape[2],
                        final_conv_length='auto')
#model = ShallowFBCSPNet(in_chans=in_chans, n_classes=n_classes,
#                        input_time_length=train_set.X.shape[2],
#                        final_conv_length='auto')
if cuda:
    model.cuda()


from braindecode.torch_ext.optimizers import AdamW
import torch.nn.functional as F
optimizer = AdamW(model.parameters(), lr=1*0.01, weight_decay=0.5*0.001) # these are good values for the deep model
#optimizer = AdamW(model.parameters(), lr=0.0625 * 0.01, weight_decay=0)
model.compile(loss=F.nll_loss, optimizer=optimizer, iterator_seed=1,)
print("INFO : Learning Rate: {}".format(0.0625 * 0.01))
print("INFO : Epochs: {}".format(50))
print("INFO : Batch Size: {}".format(8))

# Run the training
# model.fit(train_set.X, train_set.y, epochs=30, batch_size=64, scheduler='cosine',
model.fit(train_set.X, train_set.y, epochs=50, batch_size=8, scheduler='cosine',
         validation_data=(valid_set.X, valid_set.y),)


print(model.epochs_df)

# Evaluation
test_set = SignalAndTarget(X[(trainingSampleSize + valudationSampleSize):], y=y[(trainingSampleSize + valudationSampleSize):])

eval = model.evaluate(test_set.X, test_set.y)
print(eval)

try:
    print(model.predict(test_set.X))
except:
    print("predict method failed")
try:
    print(model.predict_classes(test_set.X))
except:
    print("predict_classes method failed")


