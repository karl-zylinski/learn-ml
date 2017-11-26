from sklearn import datasets

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import pprint
import matplotlib.pyplot as plt
import shutil
from PIL import Image, ImageOps
import os
from os.path import isfile, join
import glob

iw = 32

def generate_set(folder):
    source_images = list()
    targets = list()
    data = list()
    images = list()

    for n in range(0, 10):
        image_names = [f for f in os.listdir(join(folder, str(n))) if isfile(join(folder, str(n), f))]
        for f in image_names:
            unscaled_image = Image.open(join(folder, str(n), f))
            inverted_image = ImageOps.invert(unscaled_image).convert('L')
            source_images.append(inverted_image.resize((iw, iw), Image.BOX))
            unscaled_image.close()
            targets.append(n)

    for src in source_images:
        original_pixels = list(src.getdata())
        width, height = src.size
        resized_data = [0] * iw * iw
        resized_image = [0] * iw

        for i in range(len(original_pixels)):
            resized_data[i] = round((original_pixels[i]/128))

            if i % width == (width - 1):
                y = i//width
                resized_image[y] = resized_data[width*y:width*y+width]

        data.append(resized_data)
        images.append(resized_image)

    return data, targets, images

learn_data, learn_targets, learn_images = generate_set('to_learn')
clf = MLPClassifier()
clf.fit(learn_data, learn_targets)

predict_data, predict_target, predict_images = generate_set('to_predict')
result = clf.predict(predict_data)

fail = 0
success = 0
save_images = True

if os.path.exists('failures'):
    files = glob.glob('failures/*')
    for f in files:
        os.remove(f)
else:
    os.makedirs('failures')

if os.path.exists('successes'):
    files = glob.glob('successes/*')
    for f in files:
        os.remove(f)
else:
    os.makedirs('successes')

for x in range(0, len(predict_target)):
    if result[x] != predict_target[x]:
        if save_images:
            plt.figure(1, figsize=(3, 3))
            plt.imshow(predict_images[x], cmap=plt.cm.gray_r, interpolation='nearest')
            plt.savefig('failures/guess_' + str(result[x]) + '_is_' + str(predict_target[x]) + '_id_ ' + str(x) + '.png')
        fail = fail + 1
    else:
        if save_images:
            plt.figure(1, figsize=(3, 3))
            plt.imshow(predict_images[x], cmap=plt.cm.gray_r, interpolation='nearest')
            plt.savefig('successes/guess_' + str(result[x]) + '_is_' + str(predict_target[x]) + '_id ' + str(x) + '.png')
        success = success + 1

print("Failures: " + str(fail))
print("Successes: " + str(success))
print("Success rate: " + str(round(success/len(predict_target) * 100, 1)) + "%")