import scipy.io as sio
import numpy as np
# for subject_id in range(1,45):
    # try:
subject_id = 26
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
answer_vec = answer_vec_content.get('answer_vec')
segments = mat_contents.get('segments')
cropped_trial_length = 2560 # 10 sec at 256Hz sampling rate
cropped_trial_overlap = 1280 # 5 seconds of overlap between crops
    
np.set_printoptions(suppress=True)
data = np.zeros((1,62,cropped_trial_length + 1)) # XXXXXXXXXXXXXXXXX NEED TO ADD DURATION OF TRIAL AS ANOTHER PARAM
labels = np.zeros((1))
for i in range(0,36):
    if answer_vec[i] == 0: #wrong answer
        print("Q{}: skipping due to wrong answer".format(i))
        continue 
    timeToAnswerQuestion = respMat[i][3][0][0]
    currQuestionDuration = len(segments[i][0][0]) # number of timestamps in original recording
    last_crop_start = 0
    numOfCropsForCurrQuestion = 0
    for k in range(0,cropped_trial_length):
        if last_crop_start + cropped_trial_length >= currQuestionDuration:
            print("Q{}: last crop start {}, cropped trial length {}, curr question duration {}. breaking".format(i, last_crop_start, cropped_trial_length, currQuestionDuration))
            break
        numOfCropsForCurrQuestion += 1
        for j in range(0,62):
            currCrop = np.zeros((1,62,cropped_trial_length + 1))
            currCrop[0,j,k] = segments[i][0][j,(last_crop_start + k)]
            currCrop[0,j,cropped_trial_length] = timeToAnswerQuestion
        last_crop_start += cropped_trial_overlap
        data = np.concatenate((data, currCrop), axis=0)
        labels = np.concatenate((labels, np.array([i/12])), axis=0)
        print("Q{}: crop should receive label {} - received label {}".format(i, int(i/12), labels[-1]))
    print("Q{}: number of crops {}".format(i, numOfCropsForCurrQuestion))

labels = labels.astype(np.int64)
print(labels.shape)
print(data.shape)
np.save("NirDataset/croppedDataForSubject{}".format(subject_id), data[1:,:,:])
np.save("NirDataset/croppedLabelsForSubject{}".format(subject_id), labels[1:])
print("finished saving data for subject {}".format(subject_id))
    # except:
    #     print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    #     print("YYYYYYYYYY FAILED DATA PROCESSING FOR SUBJECT {} YYYYYYYYYY".format(subject_id))
    #     print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
