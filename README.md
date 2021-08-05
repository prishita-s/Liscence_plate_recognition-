# Liscence_plate_recognition

Liscence Plate Recognition using opencv and Optical Character Recognition API. 

This was a team project, created by me and @reetsethi in our 2nd year of B.Tech.

## APPROACH AND WORKING

There are three main steps to consider here.
* The first one is getting the data from the given JSON file. 
* The second one is creating a usable CSV from it. 
* The third one is creating and training a deep CNN for license plate detection. Keras is used to make the CNN part easier.

After importing all the libraries required, get into the working of the project. 
The dataset which is in the form of a .json file is first accessed. Python Language has a built-in module for JSON files. The .json file is then converted to a dataset using the pandas library in python. Since the images are in the form of URLs in the created data frame and cannot be accessed directly from there.

An unstable CSV file is created from the .json file to ease access to the images of the dataset. The informations that we recorded in the .csv file from the .json file are - image_width, image_height, x and y coordinates of top left corner and x and y coordinates of bottom right corner of the bounding box ([top_x, top_y, bottom_x, bottom_y]). 

Some of the images are in the GIF format. So, before saving the images, they are converted to the JPEG format images with three (RGB) channels by using the PIL.Image module.

We the data is split into two with a batch size of 32 images. One for training (80% of the data) and one for validation/testing (20% of the data) during training. Testing is important to see if the model overfits the training data.

## The Model

A convolutional neural network is created, having 8 convolutional layers with 4 max pool layers and a fully connected network with 2 hidden layers.

A Sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor. We created a sequential convolution model with 6 layers. 

In sequential convolution layers when we go through a Conv. layer, the output of the first Conv. layer becomes the input of the 2nd Conv. Layer. 
However, when we reach the 2nd Conv. layer, the input is the activation map(s) that result from the first layer. So each layer of the input basically describes the locations in the original image for where certain low-level features appear.

Now when you apply a set of filters on top of that (pass it through the 2nd conv. layer), the output will be activations that represent higher-level features. Types of these features could be semicircles (a combination of a curve and straight edge) or squares (a combination of several straight edges). 

As you go through the network and go through more conv. layers, you get activation maps that represent more and more complex features. By the end of the network, you may have some filters that activate when there are letters in the image.

Here is the snippet of our code for reference.




The model has the following layers:

The model is then trained using stochastic gradient descent and requires an appropriate loss function when designing and configuring your model. There are many loss functions to choose from or even what a loss function is and the role it plays when training a neural network. The Adam loss function is used here, to optimize the weights and mean squared error. 

### The Adam Optimiser

Adam is an adaptive learning rate method, which means it computes individual learning rates for different parameters. Its name is derived from adaptive moment estimation, and the reason it’s called that is that Adam uses estimations of first and second moments of the gradient to adapt the learning rate for each weight of the neural network. The N-th moment of a random variable is defined as the expected value of that variable to the power of n. More formally:

Where, m — moment, X -random variable.

Adam to optimize the weights and mean squared error as our loss function.


## Training

The Keras deep learning library includes three separate functions that can be used to train your own models: .fit, .fit_generator, .train_on_batch. All three of these functions accomplish the same task, but each follows a different method of doing it. In our model, we have used .fit_generator.

The .fit_generator  function accepts the batch of data, performs backpropagation, and updates the weights in our model. This process is repeated until we have reached the desired number of epochs.

The Keras data generator is meant to loop infinitely — it should never return or exit. Since the function is intended to loop infinitely, Keras has no ability to determine when one epoch starts and a new epoch begins. Therefore, we compute the steps_per_epoch  value as the total number of training data points divided by the batch size. Once Keras hits this step count it knows that it’s a new epoch.
Step Size (steps_per_epoch)  = Number of elements / Batch Size
In our case, the Number of elements = 232 
                     Batch Size = 32
                     Step Size = 7.2 (approx 7)

Here is the extract from the code that we’ve used to train our dataset.


### Model Loss

The model's success over the validation data is almost 80%. However, we can see that from the above figure, the training has stopped after the 30th epoch. This may occur because of the low number of training samples, or my model is not capable of learning such data. 


## Testing

The remaining 20% of the dataset undergoes testing 




The detected license plates were as follows:

                

                      


## Text Recognition using OCR API

A simple and basic approach is applied to recognize the text on the license plate. A glimpse of the code is here:







The basic concept is that a request is sent to the API using a request .post() and an API-key which when receives the request i.e. an encoded image, processes the encoded image and returns the encoded result. This encoded result is decoded. The json.loads() method is used to make it easier to parse the results. Now we are with our desired results.
