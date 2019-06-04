# import scipy.io as sio
import numpy as np
# mat_contents = sio.loadmat('Subject_001_Main_Exp_Segmented_shaked.mat')

# print(type(mat_contents))
# print(len(mat_contents))
# print(mat_contents.get('segments'))
# segments = mat_contents.get('segments') #subject1 data
# print(type(mat_contents.get('segments'))) # np.ndarray

# print(segments.shape) 
# print(type(segments[1])) #np.ndarray
# print(segments[1][0].shape)

# print(segments[1][0].shape)
# maxlenSubject1 = 128822
# print("XXXXXXXXXXXXXXXXXXXXXX")
# for i in range(1,36):
#     print('question ' + str(i))
#     print(segments[i][0].shape) #data for question i -> 62 electrodes X timestamps
#     print(type(segments[i][0])) #ndarray
    
# np.set_printoptions(suppress=True)
# Z = np.zeros((36,62,maxlenSubject1))
# for i in range(0,36):
#     timestampsForQuestioni = len(segments[i][0][0]) # number of columns
#     for j in range(0,62):
#         for k in range(0,timestampsForQuestioni):
#             Z[i,j,k] = segments[i][0][j,k]



# print(Z.shape)
# np.save("nparraysave1", Z)
print("loading data for subject 1")
X = np.load("experiment\\NirDataset\\croppedDataForSubject1.npy")
y = np.load("experiment\\NirDataset\\croppedLabelsForSubject1.npy")
for i in range(2,20):
    if (i == 6) or (i == 14) or (i == 26):
        continue
    print("loading data for subject {}".format(i))
    Xtmp = np.load("experiment\\NirDataset\\croppedDataForSubject{}.npy".format(i))
    ytmp = np.load("experiment\\NirDataset\\croppedLabelsForSubject{}.npy".format(i))
    X = np.concatenate((X, Xtmp), axis=0)
    y = np.concatenate((y, ytmp), axis=0)
    print(X.shape)
    print(y.shape)
# print("INFO : SUBJECTS NUM : 41 (44 without 6,14,26)")
print("INFO : SUBJECTS NUM : 18 (20 without 6,14)")

# print(Z.shape)

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

from sklearn.utils import shuffle

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
model = ShallowFBCSPNet(in_chans=in_chans, n_classes=n_classes,
                        input_time_length=train_set.X.shape[2],
                        final_conv_length='auto')
if cuda:
    model.cuda()


from braindecode.torch_ext.optimizers import AdamW
import torch.nn.functional as F
#optimizer = AdamW(model.parameters(), lr=1*0.01, weight_decay=0.5*0.001) # these are good values for the deep model
optimizer = AdamW(model.parameters(), lr=0.0625 * 0.01, weight_decay=0)
model.compile(loss=F.nll_loss, optimizer=optimizer, iterator_seed=1,)
print("INFO : Learning Rate: {}".format(0.0625 * 0.01))
print("INFO : Epochs: {}".format(300))
print("INFO : Batch Size: {}".format(16))

# Run the training
# model.fit(train_set.X, train_set.y, epochs=30, batch_size=64, scheduler='cosine',
model.fit(train_set.X, train_set.y, epochs=300, batch_size=16, scheduler='cosine',
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


