This is a [Keras](https://keras.io/) implementation of Regularized CapsNet with ResNet.

To know more about our proposed model, please refer to the [original paper](https://www.sciencedirect.com/science/article/abs/pii/S0031320321000388)

**************************************************************************************************************************************************
Run "CapsNet.py" for the [basic CapsuleNetwork model](https://papers.nips.cc/paper/6975-dynamic-routing-between-capsules.pdf).


**************************************************************************************************************************************************
Run "RegCapsNet.py" for the Regularized version of CapsuleNetwork model.


**************************************************************************************************************************************************
To run our "RegCapsNet Conjugate with ResNet" follow the below steps:

1- Run "mainResNet.py" for the training model using ResNet-18.
The best model will save in the folder "ResNetModels"

2- Run "ExatrctResNetFeatures.py" for extracting ResNet features.
This code loads the best model from folder "ResNetModels" (in step 1) 
and then extracts train and test features in a specific layer number. 
Features will save in the folder "ResNetFeatures"

3- Run "RegCapsNetResNet.py" aiming signatures classification. This file
used features of step 2 that saved in folder "ResNetFeatures" as input data.
**************************************************************************************************************************************************

Good luck




# Cite information
Mahdi Jampour, Saeid Abbaasi, Malihe Javidi, CapsNet Regularization and its Conjugation with ResNet for Signature Identification, Pattern Recognition, 2021, 107851, https://doi.org/10.1016/j.patcog.2021.107851.
