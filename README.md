# SplitNN for vertically partitioned data

This repository is the extension for OpenMined' s PySyft tutorials. The main focus of this repository is  building the SplitNN  architecture for vertically partitioned data. These configurations allow for multiple institutions holding different modalities of data to learn distributed models without data sharing. As a concrete example, we walk through the case where radiology centers collaborate with pathology test centers and a server for disease diagnosis. As the radiology centers holding imaging data modalities train a partial model up to the cut layer. In the same way, the pathology test center having patient test results trains a partial model up to its own cut layer. The outputs at the cut layer from both these centers are then concatenated and sent to the disease diagnosis server that trains the rest of the model. This process is continued back and forth to complete the forward and backward propagations to train the distributed deep learning model without sharing each other's raw data.

# Distribute.py
As for training these SplitNN architectures on MNIST, CIFAR, etc., we first need to split each image or data point among several workers. For that,  we will be partitioning each image and send different parts of the image to different workers.
Firstly, we must define the structure of the set of pointers which points towards the set of training data batches at different data holder's locations.

Let say, we have three data holders:- Alice, Bob, Claire. 
Each of them has their data managed into a list of batches of the same size.
```ruby
Alice_data-> [alice_batch1, alice_batch2, alice_batch3,...]
Bob_data-> [bob_batch1, bob_batch2, bob_batch3,...]
Claire_data-> [claire_batch1, claire_batch2, claire_batch3,...]
```
At the central server, the pointers to the list of data batches of each client/data holder are structured into a dictionary (say Data_pointer).
- ”Data_pointer” is a dictionary in which every,
(key, value) = (id of the data holder, a pointer to the list of batches at that data holder).
