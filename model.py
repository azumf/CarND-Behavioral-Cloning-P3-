# import prerequisites
import os
import cv2
import csv
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# I used the generator object as proprosed in the lession to preprocess the data on the fly
# instead of storing everything to the memory
samples = []
#using a reduced version of the drinving log

with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile, skipinitialspace=True, delimiter=',')
    for line in reader:
        samples.append(line)
del samples[0]

# split data
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

from sklearn.utils import shuffle

# generator object
from sklearn.utils import shuffle

def generator(samples, batch_size=32):
    num_samples = len(samples)
    correction_value = 0.1
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                for row in range(3):
                    #row 1,2,3 in csv --> mid, left, right           
                    # getting path of image
                    if "2017" in batch_sample[row]:
                        path_img = 'data/IMG/' + batch_sample[row].split('\\')[-1]
                    else:
                        path_img = 'data/IMG/' + batch_sample[row].split('/')[-1]
                    # Save the image in var
                    img = cv2.imread(path_img)
                    img_aug = cv2.flip(img,1)
                    
                    
                    angle = float(batch_sample[3])
                    angle_aug = angle * (-1)
                    
                    # left images
                    if row == 1:
                        # Steer right
                        angle += correction_value
                        angle_aug -= correction_value
                    # right images
                    elif row == 2:  
                        # Steer left
                        angle -= correction_value
                        angle_aug += correction_value
                    # Add the new image and angle to the list
                    images.append(img)
                    angles.append(angle)   
                    images.append(img_aug)
                    angles.append(angle_aug)
            X_train = np.array(images)
            y_train = np.array(angles)
            
            yield shuffle(X_train, y_train)

# model
from keras.models import Model, Sequential
from keras.layers import Cropping2D, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Conv2D
from keras.layers.core import Dense, Activation, Flatten

crop_top = 65
crop_bot = 20

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# architecture
model = Sequential()

model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((crop_top, crop_bot), (0, 0))))

# Layer 1 - Convolutional
model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
model.add(Dropout(1))

# Layer 2 - Convolutional
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
model.add(Dropout(1))

# Layer 3 - Convolutional
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
model.add(Dropout(1))

# Layer 4 - Convolutional 
model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
model.add(Dropout(1))

# Layer 5 - Convolutional
model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
model.add(Dropout(0.2))

# Layer 6 - Flatten
model.add(Flatten())

# Layer 7 - Fully Connected
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# Layer 8 - Fully Connected
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# Layer 9 - Fully Connected
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# Layer 10 - Fully Connected
model.add(Dense(1))

# train the model
model.compile(loss='mse', optimizer='adam')
multiplicator = 3
# with augmented images
#multiplicator = 6


history_object = model.fit_generator(train_generator, samples_per_epoch= len(train_samples)*multiplicator, 
                                     validation_data=validation_generator, nb_val_samples=len(validation_samples), 
                                     nb_epoch=3, verbose = 1)

print(history_object.history.keys())
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])

model.save('model.h5')

print('Model Saved!')