import os
import cv2
import numpy as np
from random import shuffle
import tflearn
from matplotlib import pyplot as plt
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation


# tf.__version__, np.__version__


# Create folder

def create_label(image_name):
    if image_name.__contains__('A'):
        return np.array([1, 0, 0, 0, 0])
    elif image_name.__contains__('B'):
        return np.array([0, 1, 0, 0, 0])
    elif image_name.__contains__('C'):
        return np.array([0, 0, 1, 0, 0])
    elif image_name.__contains__('D'):
        return np.array([0, 0, 0, 1, 0])
    elif image_name.__contains__('E'):
        return np.array([0, 0, 0, 0, 1])


DataPath = r"D:\4th year first semester\Signature-Object-Detection\Handwritten Signature Identification\Data"

IMG_SIZE = 200


def extractData(path):
    train_images = []
    test_images = []

    Personsfolder = os.listdir(path)
    for person in Personsfolder:
        personPath = os.path.join(path, person)
        train_testOfPerson = os.listdir(personPath)

        if train_testOfPerson[1] == 'Train':
            trainPath = os.path.join(personPath, train_testOfPerson[1])
            trainImgNames = os.listdir(trainPath)
            for i in trainImgNames:
                if 'SigVerificationTrainLabels' not in i:
                    img = cv2.imread(os.path.join(trainPath, i), 0)
                    img_data = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    img_normalized = img_data / 255.0

                    train_images.append([np.array(img_normalized), create_label(i)])

        if train_testOfPerson[0] == 'Test':
            testPath = os.path.join(personPath, train_testOfPerson[0])
            testImgNames = os.listdir(testPath)
            for i in testImgNames:
                if 'SigVerificationTestLabels' not in i:
                    img = cv2.imread(os.path.join(testPath, i), 0)
                    img_data = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    img_normalized = img_data / 255.0

                    test_images.append([np.array(img_normalized), create_label(i)])
    shuffle(train_images)
    shuffle(test_images)
    return train_images, test_images


train, test = extractData(DataPath)

# IMG_SIZE = 50
X_train = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_train = np.array([i[1] for i in train])

X_test = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_test = np.array([i[1] for i in test])

# IMG_SIZE = 50

# Real-time data preprocessing
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Real-time data augmentation
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)
# Model structure

network = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)
network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
network = fully_connected(network, 512, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 5, activation='softmax')
network = regression(network, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.001)

model = tflearn.DNN(network, tensorboard_verbose=3)
if os.path.exists('model.tfl.meta'):
    model.load('./model.tfl')
else:
    model.fit(X_train, y_train, n_epoch=10, show_metric=True, validation_set=0.2)
    model.save('model.tfl')

predictions = model.predict(X_test)

# print(predictions)

# pred = []
# pred_dict = {}
# pred_dict_list = []
# for prediction in predictions:
#     max_val = np.argmax(prediction)
#     pred.append(max_val)
#
#     if max_val == 0:
#         pred_dict[max_val] = 'Person A'
#     elif max_val == 1:
#         pred_dict[max_val] = 'Person B'
#     elif max_val == 2:
#         pred_dict[max_val] = 'Person C'
#     elif max_val == 3:
#         pred_dict[max_val] = 'Person D'
#     elif max_val == 4:
#         pred_dict[max_val] = 'Person E'
#     pred_dict_list.append(pred_dict[max_val])
# print(pred)
# print(pred_dict_list)
#
#
#
# act = []
# for prediction in y_test:
#     max_val = np.argmax(prediction)
#     act.append(max_val)
# print(act)
#
# c = 0
# for i in range(len(pred)):
#     if pred[i] == act[i]:
#         c += 1
# print("Test acc = " + str((c / len(pred) * 100)) + "%")



# Testing Phase
IMG_SIZE = 200
path = r'D:\4th year first semester\Signature-Object-Detection\Handwritten Signature Identification\TestData'

test_pics = os.listdir(path)
test_img = []
test_v = []

for pic in test_pics:
    img = cv2.imread(os.path.join(path, pic), 0)
    img_data = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_normalized = img_data / 255.0
    test_img.append([np.array(img_normalized)])
    test_v.append([img])
x_test = np.array([i for i in test_img]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)


pred = model.predict(x_test)

# print(pred)

pred1 = []
pred_dict1 = {}
pred_dict_list1 = []
for prediction in pred:
    max_val = np.argmax(prediction)
    pred1.append(max_val)

    if max_val == 0:
        pred_dict1[max_val] = 'Person A'
    elif max_val == 1:
        pred_dict1[max_val] = 'Person B'
    elif max_val == 2:
        pred_dict1[max_val] = 'Person C'
    elif max_val == 3:
        pred_dict1[max_val] = 'Person D'
    elif max_val == 4:
        pred_dict1[max_val] = 'Person E'
    pred_dict_list1.append(pred_dict1[max_val])
print(pred1)

print(pred_dict_list1)
