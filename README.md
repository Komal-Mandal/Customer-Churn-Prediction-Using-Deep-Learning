# Customer-Churn-Prediction-Using-Deep-Learning


# Churn Prediction Using Deep Learning

This project is a hands-on learning exercise to predict customer churn using deep learning with TensorFlow and Keras. It was developed on Kaggle to understand how to work with artificial neural networks (ANNs), specifically using the Sequential model in Keras.
## Documentation

 Table of content

#Introduction

#Dataset

#Project Outline

#Data Preprocessing

#Model Architecture

#Model Training

#Evaluation

#Installation

#Usage

#Conclusion


Introduction

Customer churn prediction helps businesses understand which customers may leave so they can take preventive actions. In this project, i  build a neural network to predict whether a customer will churn (leave the service) based on their account and demographic information. i explore TensorFlow and Keras through a deep learning model built on Kaggle.

Dataset

Source: The dataset is available in Kaggle’s Churn Modeling dataset.

Content: It has details of 10,000 bank customers, including features like CreditScore, Geography, Gender, Age, Balance, IsActiveMember, and a target variable Exited (indicating if the customer churned).

Project Outline
Data Exploration: i check data distribution, feature types, and identify any missing values.
Data Preprocessing: Preparing the data for training by encoding categorical values, scaling, and splitting the dataset.

Model Building: Using Keras’s Sequential model to build an ANN with input, hidden, and output layers.

Model Training: Training the model on Kaggle using the processed data.

Evaluation: Checking model performance using accuracy, precision, recall, and F1-score.

Data Preprocessing

Steps:

1.Encoding Categorical Variables: Converting Geography and Gender to numerical values.

2.Feature Scaling: Scaling all numerical values for consistent model performance.

3.Data Splitting: Dividing the dataset into training and testing sets.

Model Architecture

The model is built using the Keras Sequential API with the following structure:

1.Input Layer: Accepts the processed features.

2.Hidden Layers: Several dense layers with ReLU activation, allowing the network to learn complex patterns.

3.Output Layer: A single neuron with a sigmoid activation, giving a probability score for churn prediction.

Model Diagram

  Input Layer

                            |
               Dense (Hidden Layer 1, ReLU)

                            |
               Dense (Hidden Layer 2, ReLU)

                            |
               Dense (Hidden Layer 3, ReLU)

                            |
                 Output Layer (Sigmoid)


1.Activation Functions: ReLU for hidden layers, Sigmoid for the output layer.

2.Loss Function: Binary cross-entropy to optimize binary classification.

3.Optimizer: Adam optimizer for efficient training.

Model Training

Training Details:

1.Epochs: The number of times the model sees the entire dataset during training.

2.Batch Size: Number of samples processed before updating the model weights.

3.Early Stopping: We monitor the model’s validation performance to prevent overfitting.


Evaluation
The model’s performance is evaluated based on:

1.Accuracy: The percentage of correct predictions.

2.Precision and Recall: Balances false positives and false negatives.

3.F1 Score: The harmonic mean of precision and recall for a balanced evaluation.

Conclusion

This project demonstrates the basics of building a deep learning model to predict customer churn using Keras and TensorFlow. It provides insights into data preprocessing, neural network design, and model evaluation metrics. This project can be extended with additional data sources or more complex architectures for improved performance.


## Image

<img src= "(![alt text](<churnk prediction.webp>))"



## Acknowledgements

I would like to express my gratitude to CampusX and Krish Naik for their insightful tutorials on machine learning and deep learning. Their YouTube content played a key role in helping me understand and implement the concepts used in this project.


## 🔗 Links
Kaggle Link
(https://www.kaggle.com/code/mandalkumkum/notebook2e6a5e4f95)



## 🛠 Skills
HTML,CSS,javaScript,jupyter notebook,Numpy,Pandas
Matplotlib,ScikitLearn, Keras,Tensorflow


## Other Common Github Profile Sections
👩‍💻 I'm currently working on AI/Ml

🧠 I'm currently learning Deep
Learning


📫 How to reach me
https://github.com/Komal-Mandal

https://www.linkedin.com/in/komal-mandal-b04006259








