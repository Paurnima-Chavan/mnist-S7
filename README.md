
# Handwritten Digit Recognition using Convolutional Neural Networks with Pytorch

The MNIST dataset comprises 70,000 grayscale images of handwritten digits, each measuring 28 x 28 pixels and annotated by humans. It is a subset of a larger dataset made accessible by NIST, the National Institute of Standards and Technology. In this example, we will develop a handwritten digit recognizer utilizing a convolutional neural network (CNN) trained on the MNIST dataset, employing the PyTorch framework.

<p align="center">    
    <img width="500" aling="right" src="https://github.com/Paurnima-Chavan/MNIST-pytorch/blob/main/imgs/handwriiten.png?raw=true" />
</p>


## Documentation

This documentation provides an overview of the PyTorch model used for classifying handwritten digits from the MNIST dataset. The model architecture consists of convolutional neural networks (CNNs) and fully connected layers, which are designed to achieve accurate digit recognition.

## Code organization

Code organization in this project is structured into three files.

src

 &nbsp;&nbsp;&nbsp;&nbsp; `models.py` 

 &nbsp;&nbsp;&nbsp;&nbsp; `utils.py`

 &nbsp;&nbsp;&nbsp;&nbsp; `dataset.py`

`S7_Step1.ipynb`

`S7_Step2.ipynb`

`S7_Step3.ipynb`

`S7_Step4.ipynb`

The **"models.py"** file consists of four distinct models. The **"utils.py"** file contains the code responsible for training, testing, and generating performance graphs. In **"dataset.py,"** you will find the necessary code for loading the MNIST dataset.

Finally, the notebooks **"S7_Step1.ipynb,"** "**S7_Step2.ipynb**," "**S7_Step3.ipynb**," and "**S7_Step4.ipynb**" are dedicated to the execution and experimentation on different models.

This separation allows for the modular and organized development of the project components.

## Goal

The goal is to reach a Test Accuracy of 99.4% within 15 Epochs or less, using fewer than 8,000 Parameters, on the MNIST dataset.

## Step 1: The Setup 
### **Target**:
1.	Ensure the correct setup.
2.	Define the necessary transformations.
3.	Configure the data loader.
4.	Establish the foundation of the code.
5.	Implement the basic training and testing loop.
6.	Establish a robust skeleton structure, minimizing subsequent modifications if possible.
   
### **Results**:
1.	Parameters: 190K -- 190,442
2.	Best Training Accuracy: 98.57
3.	Best Test Accuracy: 98.95
   
  	![image](https://github.com/Paurnima-Chavan/mnist-S7/assets/25608455/45ed2b83-45db-472f-a5c3-12a5e1c14218)
  	
**The final few epochs:**

![image](https://github.com/Paurnima-Chavan/mnist-S7/assets/25608455/05167e8b-5b3e-4159-8f72-17ab1ffa7bfb)

### **Analysis**:
1.	Heavy Model for such a problem

## Step 2: The Skeleton
### Target:

Ensure the fundamental structure is correct. We will make an effort to minimize alterations to this framework.
### Results:

1. Parameters: 18K
2. Best Training Accuracy: 99.17
3. Best Test Accuracy: 99.35

![image](https://github.com/Paurnima-Chavan/mnist-S7/assets/25608455/20ad695c-fee2-4afc-aa34-dcc79eff1f21)

**The final few epochs:**

![image](https://github.com/Paurnima-Chavan/mnist-S7/assets/25608455/3ffbb266-c295-463a-80ce-ae51606a3eb0)

### Analysis:

1. The model demonstrates promise, but it requires further optimization to reduce its overall weight.
2. No signs of over-fitting have been observed, indicating that the model has the potential to perform even better with additional training or adjustments.

## Step 3: The Batch Normalization
### Target:
1.	Add Batch-norm to increase model efficiency.
### Results:
1.	Parameters: 6.2K -- 6,270
2.	Best Training Accuracy: 99.03
3.	Best Test Accuracy: 99.34
   
   ![image](https://github.com/Paurnima-Chavan/mnist-S7/assets/25608455/d6e04895-3628-4265-9c39-069bd77f10b4)

**The final few epochs:**

![image](https://github.com/Paurnima-Chavan/mnist-S7/assets/25608455/a1319e09-fe7b-4853-b7ea-4c246f72145d)

### Analysis:
1.	As we have decreased the capacity of the model, it is anticipated that there will be a decline in performance.
2.	To further enhance the model, it is necessary to augment the model capacity and adjust other relevant parameters.

## Step 4: Increase the Capacity, Correct MaxPooling Location m11
### Target:
1.	Enhance the model capacity by incorporating additional layers at the end.
2.	Correct the position of max pooling in the model architecture.
3.	Fine-tune the learning rate to optimize model performance.
4.	Fine-tune the learning rate to optimize model performance.	
### Results:
1.	Parameters: 7K -- 7,074
2.	Best Training Accuracy: 98.94
3.	Best Test Accuracy: 99.46
   
   ![image](https://github.com/Paurnima-Chavan/mnist-S7/assets/25608455/7c4390ab-7871-4352-9ada-3e9500396526)
   
**The final few epochs:**

   ![image](https://github.com/Paurnima-Chavan/mnist-S7/assets/25608455/ce06c07f-4fca-454a-884d-419d900f6bd1)


### Analysis: 
1.	The implemented changes proved to be effective as the model achieved a target accuracy of 99.4% after 15 epochs of training.
