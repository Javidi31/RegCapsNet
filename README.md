# RegCapsNet with ResNet - Keras/Tensorflow

This is a [Keras](https://keras.io/) and [Tensorflow](https://www.tensorflow.org/) implementation of Regularized CapsNet with ResNet.

To know more about our proposed model, please refer to the [original paper](https://www.sciencedirect.com/science/article/abs/pii/S0031320321000388)

**************************************************************************************************************************************************
Run "[CapsNet.py](https://github.com/Javidi31/RegCapsNet/blob/main/CapsNet.py)" for the [basic CapsuleNetwork model](https://papers.nips.cc/paper/6975-dynamic-routing-between-capsules.pdf).



**************************************************************************************************************************************************
Run "[RegCapsNet.py](https://github.com/Javidi31/RegCapsNet/blob/main/RegCapsNet.py)" for the Regularized version of CapsuleNetwork model.



**************************************************************************************************************************************************
To run our "RegCapsNet Conjugate with ResNet" follow the below steps:

1- To prepare the dataset, download it from the following and then run "[Preparing.m](https://github.com/Javidi31/RegCapsNet/blob/main/Preparing.m)" 

2- Run "[mainResNet.py](https://github.com/Javidi31/RegCapsNet/blob/main/mainResNet.py)" for the training model using ResNet-18.
The best model will be saved in the folder "ResNetModels"

3- Run "[ExatrctResNetFeatures.py](https://github.com/Javidi31/RegCapsNet/blob/main/ExtractResNetFeatures.py)" for extracting ResNet features.
This code loads the best model from folder "ResNetModels" (in step 1) 
and then extracts train and test features in a specific layer number. 
Features will be saved in the folder "ResNetFeatures"

4- Run "[RegResCapsNet.py](https://github.com/Javidi31/RegCapsNet/blob/main/RegResCapsNet.py)" aiming signatures classification. This 
file used features of step 2 (which are saved in folder "ResNetFeatures") 
as input data.

**************************************************************************************************************************************************
# Dataset

1- Cedar dataset is Available at (http://www.cedar.buffalo.edu/NIJ/data/signatures.rar) (a number of samples of original and pre-processed dataset are available in the Datasets folder).


**************************************************************************************************************************************************
For more information about loading dataset or setting the parameters, please refer to utilities folder.
**************************************************************************************************************************************************


Good luck




# Cite information
Mahdi Jampour, Saeid Abbaasi, Malihe Javidi, 

CapsNet Regularization and its Conjugation with ResNet for Signature Identification, 

Pattern Recognition, 2021, 107851, https://doi.org/10.1016/j.patcog.2021.107851.
