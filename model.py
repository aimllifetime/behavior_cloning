import csv
import cv2
import numpy as np
import tensorflow as tf
import sys
import os

from keras.models import load_model
# update Keras to latest version 2.0.1: pip install keras --upgrade

# add command line processing to read in the IMG and previously trained hd5 file
# example:
# python model.py -d "./data;./revers;./my_data"   <= start fresh training without loading any h5
# python model.py -d "./data" -m model.h5   <= load previously trained model.h5 and train with ./data
# python model.py -d "./data" -m model.h5 -p 0.3 -e 50 <= reduce image with measurement of 0 by 30 percentage. run epoch of 50
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-d','--dir', required=True, help='IMG dir where data is collected: "dir1;dir2;dir3"')
parser.add_argument('-m', '--md5', help='Previously trained model is used. if this is blank, training starts from stratch')
parser.add_argument('-p', '--percentage', type=float, default='0.3', help='percentage of repeative image that has measurement of 0')
parser.add_argument('-e', '--epoch', type=int, default='50', help='number of epochs to run')
args = parser.parse_args()

print(args)

lines = []
processed_lines = []

import re
def pre_process_csv(dirs):
    """this function read in all directories specified by -d argument.
    the directories is seperated by ';''
    Input:   dirs      <= couple of directories each stores particular captures
    Output:  l_lines[] <= each entry is original csv file.
                          concatenate all csv to single result store in lines.
                          replace the path in the original csv file to become local relative path.
    """
    l_lines = []
    for df in dirs:
        if(os.path.exists(df)):
            log_csv = df + '/' + 'driving_log.csv'
            print(log_csv)
            with open(log_csv) as csvfile:
                reader = csv.reader(csvfile)
                for line in reader:
                    if(re.search('speed', line[6])) : # ignore the csv headline.
                        continue
                    else :
                        # substitue the path to current path for center, left and right
                        filename = line[0].split('/')[-1]
                        line[0] = df + '/IMG/' + filename
                        filename = line[1].split('/')[-1]
                        line[1] = df + '/IMG/' + filename
                        filename = line[2].split('/')[-1]
                        line[2] = df + '/IMG/' + filename
                        #print(line)
                        l_lines.append(line)
    return l_lines


import random
def reduce_similar_image(lines):
    """
    This function particularly reduce the similar image with measurement of 0.
    First, it saves all image with non-zero measurement to one array l_processed_lines.
    Meanwhile, it save all image with zero measurement to another array repeat_line_0.
    then only randomly choose the defined percentage of repeat_line_0 and append
    them to l_processed_lines, where percentage is passed from command line -p []
    """
    l_processed_lines = []
    repeat_line_0 = [];

    for line in lines:
        if(float(line[3]) != 0.0):
            l_processed_lines.append(line)
        else :
            repeat_line_0.append(line)

    print(len(l_processed_lines))

    #print(len(lines))

    # randomly append only 30% of repeat_line_0
    random.shuffle(repeat_line_0)
    print(len(repeat_line_0))
    for index in range(int(len(repeat_line_0) * args.percentage)):
        l_processed_lines.append(repeat_line_0[index])

    print(len(l_processed_lines))
    return l_processed_lines


# read in the command line -d arguments, seperate each dir.
# call the pre_process_csv to have single csv buffer from different directories.
# replace the absolute path to relative path to image location.
# reduce the repeatative image of zero measurement
dirs = []
if (args.dir):
    dirs = args.dir.split(";")
    #print (dirs)

    lines = pre_process_csv(dirs)
    lines = reduce_similar_image(lines)

#print(lines)
if(len(lines) == 0):
    print("does not find any right image files")
    sys.exit()


images = []  # stores all images read them from pre-processed csv file. This also
             # includes all side-camera images with offset measurement,
             # flipped images.
measurements = []  # each image's corresponding measurement.

for line in lines[1:-1]:
    # for each line, read in the first three columns, each corresponds to center, left and right camera images.
    center_current_path = line[0]
    left_current_path   = line[1]
    right_current_path  = line[2]

    print(center_current_path)
    # read in the three image for center, left and right.
    image = cv2.imread(center_current_path)
    image_left = cv2.imread(left_current_path)
    image_right = cv2.imread(right_current_path)
    # convert the measurement to float for later processing.
    measurement = float(line[3])
    # append the centeral image without change.
    images.append(image)
    measurements.append(measurement)
    # flipped the center image as well as the measurement.
    # append the flipped image and its storage.
    image_flipped = np.fliplr(image)
    measurement_flipped = -measurement
    images.append(image_flipped)
    measurements.append(measurement_flipped)
    # adding the left side camera image. adding the measurement offset 0.15
    # append the image and measurement to storage
    correction = 0.15
    steering_left = measurement + correction
    images.append(image_left)
    measurements.append(steering_left)
    # adding the left side camera image. adding the measurement offset 0.17
    # add a bit less offset to the right image because the car seems drive more
    # toward to right side of road.
    # append the image and measurement to storage
    steering_right = measurement - correction - 0.02
    images.append(image_right)
    measurements.append(steering_right)


import cv2
import math
import numpy as np
from keras.layers import Cropping2D

X_train = np.array(images)
y_train = np.array(measurements)

print(len(X_train))
print(len(y_train))

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

def model_nvda_covnet():
    """
    using Nvida covnet architecture primarily but with some dropouts.
    First layer: using lambda to normalize the image.
    second layer: cropping the upper half image so only majority of road is only presented.
    third layer: convolutional with 24 filters and filter 5x5, stride (2,2), activation of relu.
    fourth layer: convolutional with 36 filters and filter 5x5, stride (2,2), activation of relu.
    fifth layer: convolutional with 48 filters and filter 5x5, stride (2,2), activation of relu.
    Dropout probability 0.3
    sixth layer: convolutional with 64 filters and filter 5x5, stride (1,1), activation of relu.
    seventh layer: same as sixth layer.
    eighth layer: flatten layer
    dropout probability 0.3
    ninth layer: densely connect layer of 100
    tenth layer: densely connect layer of 50
    last layer: is output layer with one node.
    """
    model = Sequential();
    model.add(Lambda(lambda x : x/255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70,25), (0, 0))))
    model.add(Conv2D(24,(5,5),strides=(2,2),activation='relu'))
    model.add(Conv2D(36,(5,5),strides=(2,2),activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2,2))) #added
    model.add(Conv2D(48,(5,5),strides=(2,2),activation='relu'))
    model.add(Dropout(0.3))
    model.add(Conv2D(64,(3,3),activation='relu'))
    model.add(Conv2D(64,(3,3),activation='relu'))
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(100))
    #model.add(Dropout(0.5)) cause local round turn.
    model.add(Dense(50))
    model.add(Dense(1))
    return model


# if command line gives -m <*.h5>, then the file is read in for current.
# if not load previously trained model, the this is a fresh training. will use nvda covnet.
if(args.md5):
    print("load previously trained model")
    model = load_model('./model.h5')
else:
    model = model_nvda_covnet()


model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split = 0.2, shuffle= True, nb_epoch=args.epoch)

model.save('model.h5')
