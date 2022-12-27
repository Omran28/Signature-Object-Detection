import os
import csv
import random
import cv2
import numpy as np
from tensorflow.keras.applications.inception_v3 import preprocess_input

path = r"D:\4th year first semester\Signature-Object-Detection\Data"
size = 128


def Creating_Triplets(type):
    Triplets = []
    real_img_names = []
    fake_img_names = []
    personsFolder = os.listdir(path)

    for person in personsFolder:
        personPath = os.path.join(path, person)
        train_testOfPerson = os.listdir(personPath)
        image_path = os.path.join(personPath, train_testOfPerson[type])

        trainImgNames = os.listdir(image_path)
        csv_File = os.path.join(image_path, trainImgNames[-1])
        with open(csv_File, 'r') as file:
            File_Reader = csv.reader(file)
            for row in File_Reader:
                if row[0] == "image_name":
                    continue
                elif row[1] == 'real':
                    real_img_names.append(row[0])
                else:
                    fake_img_names.append(row[0])

            for i in range(len(real_img_names) - 1):
                for j in range(i + 1, len(real_img_names)):
                    anchor = (person, real_img_names[i])
                    positive = (person, real_img_names[j])
                    random_fake = random.randint(1, len(fake_img_names) - 1)
                    negative = (person, fake_img_names[random_fake])
                    Triplets.append((anchor, positive, negative))
        real_img_names = []
        fake_img_names = []
    random.shuffle(Triplets)
    return Triplets


def get_batch(triplet_list, folder, batch_size=256, preprocess=False):
    batch_steps = len(triplet_list) // batch_size

    for i in range(batch_steps + 1):
        anchor = []
        positive = []
        negative = []

        j = i * batch_size
        while j < (i + 1) * batch_size and j < len(triplet_list):
            a, p, n = triplet_list[j]
            anchor.append(read_image(a, folder))
            positive.append(read_image(p, folder))
            negative.append(read_image(n, folder))
            j += 1

        anchor = np.array(anchor)
        positive = np.array(positive)
        negative = np.array(negative)

        if preprocess:
            anchor = preprocess_input(anchor)
            positive = preprocess_input(positive)
            negative = preprocess_input(negative)

        yield [anchor, positive, negative]


def read_image(index, folder):
    img_path = os.path.join(path, index[0], folder, index[1])
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (size, size))
    image = rescale(image, -1, 1)
    return image


def rescale(img, min, max):
    img = (img - img.min()) / float(img.max() - img.min())
    img = min + img * (max - min)
    return img
