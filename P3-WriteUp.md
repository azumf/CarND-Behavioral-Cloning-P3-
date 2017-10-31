# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* model.ipynb corresponding jupyter notebook
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Visualization of the training images

#### 1. Using the images of all three cameras

For training the images from all cameras mounted on the vehicle are used.

**Left camera:**

<img src="/writeup_imgs/left.jpg " width="250"/>

**Center camera:**

<img src="/writeup_imgs/center.jpg " width="250"/>

**Right camera:**

<img src="/writeup_imgs/right.jpg " width="250"/>

#### 2. Flipped images

To better generalize, all images yielded by the generator object are flipped. That helps the model to generalize and as well adds augmented data to the training data pool.

**recorded image and flipped image**

<img src="/writeup_imgs/flip1.jpg " width="250"/> 
<img src="/writeup_imgs/flip2.jpg " width="250"/>



### Model Architecture and Training Strategy

#### 1. Description of model architecture

The lines information refer to the model.py file, not to the jupyter notebook. The model.py is just for explanation purposes.

My model consists of a convolution neural network with filter sizes of 5x5 and 3x3 respectively. The filter depths varies between 32 and 128 (model.py lines 86-131) 

The model includes non linear RELU activation function layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 89). 

By applying the Cropping2D layer, the image is cropped by 20 pixels from bottom and by 65 pixels from top. These should help to focus the feature extraction by the convolutional layers on the lane itself and not on any surroundings.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting.

First, I trained the model with sample sizes of ~40000 images per epoch and for only 3 epochs. The training loss was significantly reduced during the first epoch. The following epochs did reduce the trainig loss only in a minor fashion. However, the validation loss was lowered as well.

For better understanding and to keep track of the model performance I chose to re-train the model with a smaller sample size per epoch (~3700 samples). I will get more in detail later on.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. 


#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 134).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving and recovering from the left and right sides of the road. The recovering data was generated with the simulator tool. 
I as well flipped the images during processing by the generator object to multiply the available training data.


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to analyse the existing models like GoogLeNet or VGG as we learned it in the transfer learning lesson. It was clear that the architecture would be a stack of convolutional and fully connected layers.

Coming from my knowledge I acquired during the deep learning Nanodegree it was obvious that one or two convolutional layers won't do the job. I started with 3 convolutional layers with filter sizes from 24 to 48 but went deeper with 2 additional conv. layers to gain a more detailed feature extraction.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 

To combat the overfitting, I added dropout layers after the convolutional layers and after the fully connected layers. After the conv. layers I used a keep_prob of 0.2, after the FC layers I used 0.5 as dropout parameter.

The final step was to run the simulator to see how well the car was driving around track one. 

**There were a single spot where the vehicle had difficulties. At 0:43s in the video file, the vehicle steered to the left side of the lane. However, the model managed to steer back to the center of the lane before the driveable portion was left.**
I assume this could be optimized by recording more of these "turn back manoeuvres", especially from the left side of the lane. As well finetuning the correction factor for steering could help out. However, due to the high time demand of the training (~16h on my GPU), I was satisfied with this result and did not re-trained the model due to the mentioned required time.


At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 86-131) consisted of a convolution neural network with the following layers and layer sizes:

| Layer   |      Type      |  Parameters |
|----------|:-------------:|------:|
| 1 | conv. | f_depth = 24 f_size = (5x5) keep_prob = 1.0|
| 2 | conv. | f_depth = 36 f_size = (5x5) keep_prob = 1.0|
| 3 | conv. | f_depth = 48 f_size = (5X5) keep_prob = 1.0|
| 4 | conv. | f_depth = 64 f_size = (3X3) keep_prob = 1.0|
| 5 | conv. | f_depth = 128 f_size =  (3x3) keep_prob = 0.2|
| 6 | Flatten | None|
| 7 | Fully connected | n_neurons = 128 / keep_prob = 0.5|
| 8 | Fully connected | n_neurons = 64 / keep_prob = 0.5|
| 9 | Fully connected | n_neurons = 16 / keep_prob = 0.5|
| 10 | Fully connected | n_neurons = 1 |


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I flipped the already recorded lane driving with the cv2.flip function, see above. It was of course necessary to flip the values of the steering angle as well.

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to steer back in case of reaching the lane sides. I assume that the situation with low performance in the video is as well related to the recovery situations that I have recorded. Sometimes I did not capture the best lane back to the center.

After the collection process, I had 27.927 number of unique data points or images. By applying the cv2.flip function the dataset is doubled. With consideration of the augmented pictures as well I got a total number of 55.854 images. 

I then preprocessed this data by usage of a generator object. With this generator the data is processed on the fly. The generator yields images and corresponding angles with a batch_size of 32.

For the correction value for the steering angle (for angle values that correspond to images from the left or the right camera) I chose **0.2**. First I tried a correction value of **0.1** but it was shown that the vehicle left the lane in a curve with a high curvature.

20% of the training data was considered as validation data.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I used an adam optimizer so that manually training the learning rate wasn't necessary. A number of 15 epochs showed a proper loss reduction.

<img src="/writeup_imgs/loss_valloss.png " width="250"/>

