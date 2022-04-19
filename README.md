# CAS771 Triteaching_plus
## Running procedures
### 1. Download the whole repository as a zip file
### 2. Unzip in local and change the corresponding data_path in main.py
### 3. Install all of the library packages
### 4. Run the model on cifar10 task1 by inputing python main.py --dataset cifar10 --task task1 --n_epoch 50 --CNN deep on command line. To edit other parameters from default, add arguments on command line or edit default arguments directly in main.py
## Description
### 1. loader_for_ANIMAL10n.py and loader_for_CIFAR.py loads data of cifar10 and animal10n respectively
### 2. loss.py contains the algorithm for loss function
### 3. model.py contains a shallow CNN network and a deep CNN network.
### 4. main.py contains training and testing functions, which encapsulate the loss function. The main function constructs the CNN networks and calls training and testing functions in each epoch.
