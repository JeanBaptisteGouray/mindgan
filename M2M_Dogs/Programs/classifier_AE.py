import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.optim as optim

import utils
import models
import save
import helper

import os
import numpy as np
import time
import shutil
import collections

data_path, dataset = utils.recup_datas('classifier_AE')

print('Les datasets se trouvent a l\'emplacement :', data_path)
print('Le dataset utilise est :', dataset)

folder = '../Classifier_AE'

if not os.path.exists(folder):
    os.makedirs(folder)

if not os.path.exists(folder + '/Hyperparameters.csv'):
    shutil.copyfile('../../Hyperparameters/Hyperparameters_classifier.csv',  folder + '/Hyperparameters.csv')

os.chdir(folder)

training_folder = 'Trainings/'
# Folder where trainings are saved
if not os.path.exists(training_folder):
    os.makedirs(training_folder)

# Create the folder by day and time to save the training
folder = training_folder + time.strftime('%Y_%m_%d_%H_%M_%S')

del training_folder

print("Toutes les donnees sont enregistrees dans le dossier : " + folder)

# Select hyperparameters for the training
hyperparameters = utils.select_hyperparameters('Hyperparameters.csv')
print('Hyperparameters = ', hyperparameters)

# Save hyperparameters
save.save_hyperparameters(hyperparameters, folder)

# Add the hyparameters at the file Tested_hyperparameters.csv
save.save_tested_hyperparameters(hyperparameters)

# Hyperparameters for the training
[batch_size, num_workers, conv_dim, lr, beta1, beta2, epochs] = list(hyperparameters.values())

# Folders
checkpoint_path = folder + '/checkpoints/'

if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

# Transformation and datasets
transform = transforms.Compose([transforms.Resize(140),
                                transforms.CenterCrop(128),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
                                ])

data_path, train_loader, test_loader = utils.dataset(data_path, dataset, batch_size, transform)

nb_classes = len([f for f in os.listdir(data_path + '/test') if os.path.isdir(os.path.join(data_path + '/test', f))])

image = next(iter(train_loader))[0][0]

nb_channels = image.shape[0]
height = image.shape[1]
width = image.shape[2]

del image

print('Il y a {} classes'.format(nb_classes))
print('La taille des images est de : ({},{},{})'.format(nb_channels, height, width))

# Create the Neural networks
model = models.Pretrain_Classifier(nb_classes)

# Trainig on GPU if it's possible
train_on_gpu = torch.cuda.is_available()

if train_on_gpu:
    # move models to GPU
    model.cuda()
    print('GPU available for training. Models moved to GPU \n')
else:
    print('Training on CPU. \n')

# Define optimizer and loss function
optimizer = optim.Adam(model.classifier.parameters(),lr, [beta1,beta2])

criterion = nn.NLLLoss()

start = time.time()

# Training
test_loss_min = np.inf

for epoch in range(epochs):
    train_loss = 0.

    for images, labels in train_loader:

        # Move images and labels on GPU if we train on GPU
        if train_on_gpu:
            images = images.cuda()
            labels = labels.cuda()
        
        # Predictions by the NN
        predictions = model.forward(images)

        # Reset the gradient
        optimizer.zero_grad()

        # Compute the loss and add it to the train_loss 
        loss = criterion(predictions,labels) 
        train_loss += loss

        # One step in the gradient descent
        loss.backward()
        optimizer.step()
        
    
    # Stop the computing of the gradient
    with torch.no_grad():
        # Tur the NN in evaluation mode
        model.eval()

        test_loss = 0.
        accuracy = 0.

        for images, labels in test_loader:

            # Move images and labels on GPU if we train on GPU
            if train_on_gpu:
                images = images.cuda()
                labels = labels.cuda()
            
            # Predictions by the NN
            predictions = model.forward(images)

            # Compute the loss and add it to the test_loss
            loss = criterion(predictions,labels)
            test_loss += loss

            # Compute the accuracy of the right predictions
            ps = torch.exp(predictions)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            

    # Turn the NN on train mode        
    model.train()

    print('Epoch [{:5d}/{:5d}] | Time: {:6.0f} | Training loss: {:6.4f} | Testing loss: {:6.4f} | Accuracy: {:3.2f}% '.format(epoch+1, epochs, time.time() - start, train_loss.item()/len(train_loader), test_loss.item()/len(test_loader),accuracy/len(test_loader)*100), end='')
    
    # Save the classifier if the test_loss is better
    test_loss_min, affichage = save.save_model(test_loss.item()/len(test_loader), test_loss_min, model, checkpoint_path, train_on_gpu)

    if affichage:
        print("| Model saved !")
    else:
        print()

    save.save_log_classifier(epoch, time.time()-start, train_loss.item()/len(test_loader), test_loss.item()/len(test_loader), accuracy/len(test_loader)*100, folder)

save.save_best_classifier(test_loss_min, hyperparameters, checkpoint_path, mode='AE')