# Facial-Expression-Detection-Deep-Learning
This project contains custom expression (Happy and Sad) detection code. The model is trained using transfer learning with base model as DenseNet201.

# Functionality

* This project demonstrates use of powerful and advanced Deep Learning for Computer Vision.
* We can create our own custom facial dataset for various expressions (Happy and Sad for this project) with the method **capture_data_from_videocapture()**.
* We use transfer learning to fine tune the weights of pre-trained DenseNet201 model to learn the patterns in our custom data.
* Final output is the classified expression which our model identifies. 

![image](https://user-images.githubusercontent.com/59373491/121079423-b5731600-c7f7-11eb-9e8c-ebbb02236866.png) ![image](https://user-images.githubusercontent.com/59373491/121079504-d3407b00-c7f7-11eb-8b92-1a2ccad0de54.png)



# Packages Used

* OpenCV
* time
* Numpy
* imutils
* keras

# Steps Involved

* First step before solving any traditional Machine Learning problem or Deep Learning problem is to collect data with labels.
* **capture_data_from_videocapture()** method is used to capture your custom dataset using your default webcam.
* It uses pre-trained model whose prototxt and weight file can be downloaded from web.
* Nect we take our base model DenseNet201 for transfer learning, we remove the top layer (Fully connected layers) and keep 
our custom Dense layers with last layer having only two nodes (Happy or Sad).
* We use ImageDataGenerator to augment the captured data, it helps to increase the size of dataset which in turn helps model prevent overfitting.
* We start the training process but we keep the base model as untrainable and the best weights are restored using checkpoint. The training process stops if there are 4 consecutive iterations where validation accuracy doen't improve.
* After first training completes, we again train our model. But this time our head or the custom Dense layers are warmed up or trained, so now we let the loss propagate to the base model. Hence, we set the base model as trainable.
* We start the training process again and keep the learning rate very low.
* Once the model training completes, we use this model in **detect_expression()** to classify the expression in images.
