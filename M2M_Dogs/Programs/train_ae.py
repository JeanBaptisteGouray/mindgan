import torch
from torchvision import transforms, datasets
import torch.optim as optim
import torch.nn as nn
from torchvision import models as pretrain

import numpy as np

import time
import os
import shutil

import utils
import models
import evaluate
import save

if not os.path.exists('../checkpoints/Best_Clas_AE/Hyperparameters.txt'):
    print('Veuillez entrainer un classifieur pour AutoEncoder!!')
    exit()

epochs = 200

data_path, dataset = utils.recup_datas('AE')

folder = '../AE'
if not os.path.exists(folder):
    os.makedirs(folder)

if not os.path.exists(folder + '/Hyperparameters.csv'):
    shutil.copyfile('../../Hyperparameters/Hyperparameters_AE.csv',  folder + '/Hyperparameters.csv')

# Go to the folder AE
os.chdir(folder)

# Folder where trainings are saved
training_folder = 'Trainings/'

if not os.path.exists(training_folder):
    os.makedirs(training_folder)

# Create the folder by day and time to save the training
folder = time.strftime('%Y_%m_%d_%H_%M_%S')
folder = training_folder + folder

if not os.path.exists(folder):
    os.makedirs(folder)

print("Toutes les donnees sont enregistrees dans le dossier : " + folder)

# Select hyperparameters for the training
hyperparameters_AE = utils.select_hyperparameters_row('./Hyperparameters.csv')

# Add the hyparameters at the file Tested_hyperparameters.csv
save.save_tested_hyperparameters(hyperparameters_AE)

# Save the hyperparameters for the training
save.save_hyperparameters(hyperparameters_AE, folder)

# Hyperparameters for the training
[batch_size, num_workers, z_size, lr, beta1, beta2, latent_size] = list(hyperparameters_AE.values())

# Folders
checkpoint_path = folder + '/checkpoints/'
sample_path = folder + '/samples/'

# Make them if they don't exist
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

if not os.path.exists(sample_path):
    os.makedirs(sample_path)

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

# Parameter for the print
print_every = len(train_loader)//1

# Creation of the crtic and the generator
Encoder = models.Critic(height, width, latent_size = latent_size, mode='AE', nb_channels=nb_channels)
Decoder = models.Generator(height, width, z_size=z_size, latent_size = latent_size, mode='AE', nb_channels=nb_channels)

# Creation of the classifier which uses to compute the FId and IS
Classifier = models.Pretrain_Classifier(nb_classes)
state_dict = torch.load('../checkpoints/Best_Clas_AE/classifier.pth')
Classifier.load_state_dict(state_dict)
Classifier.eval()


# Training on GPU if it's possible
train_on_gpu = torch.cuda.is_available()

if train_on_gpu:
    # move models to GPU
    Encoder.cuda()
    Decoder.cuda()
    print('GPU available for training. Models moved to GPU. \n')
else:
    print('Training on CPU.\n')

# Optimizer for Classifier and Generator
e_optimizer = optim.Adam(Encoder.parameters(), lr, [beta1, beta2])
d_optimizer = optim.Adam(Decoder.parameters(), lr, [beta1, beta2])

criterion = nn.MSELoss()

# Initialisation if IS_max and FID_min
FID_min = np.inf

FID_mean = 0.
# Save the time of start
start = time.time()

# Training

for epoch in range(epochs):
    train_loss = 0.
    for real_images, _ in train_loader:
    
        batch_size = real_images.size(0)

        # Rescale images between -1 and 1
        real_images = utils.scale(real_images)


        # Move real_images on GPU if we train on GPU
        if train_on_gpu:
            real_images = real_images.cuda()

        reconstru = Decoder.forward(Encoder.forward(real_images))

        # Reset the gradient
        e_optimizer.zero_grad()
        e_optimizer.zero_grad()

        # Computing batch loss
        loss = criterion(reconstru,real_images)

        # Add the batch loss to epoch loss
        train_loss += loss

        # Compute gradient loss
        loss.backward()

        # Step in Gradient descent
        d_optimizer.step()
        e_optimizer.step()
        
    
    # Deactivate gradients for evaluation
    with torch.no_grad():
        # Deactivate dropout/batchnorm etc. to evaaluate the network
        Decoder.eval()
        Encoder.eval()
        test_loss = 0.
        score_fid = 0.
        for real_images, _ in test_loader:
            
            # Rescale images between -1 and 1
            real_images = utils.scale(real_images)

            # Move images on GPU if we train on GPU
            if train_on_gpu:
                real_images = real_images.cuda()
            
            # Recontruct images with autoencoder
            reconstru = Decoder.forward(Encoder.forward(real_images))

            # Computing batch loss
            loss = criterion(reconstru,real_images)

            # Add the batch loss to epoch loss
            test_loss += loss

            #Add FID score of the    batch to the FID score of the epoch
            
            score_fid += evaluate.fid(real_images,reconstru,Classifier)
    
    save.save_log_AE(epoch,time.time()-start, train_loss, test_loss, score_fid, folder)

    # Reactivate regularizers to train
    Decoder.train()
    Encoder.train()

    test_loss_min, affichage = save.save_model_test_loss(test_loss, test_loss_min, Encoder, Decoder, checkpoint_path, mode='AE')

    print('Epoch [{:5d}/{:5d}] | Time: {:.0f} | Training loss: {:6.4f} | Testing loss: {:6.4f} | FID: {:6.4f}'.format(
            epoch+1, epochs, time.time()-start, train_loss.item()/len(train_loader), test_loss.item()/len(test_loader),score_fid/len(test_loader)), end=' ')
    
    if affichage:
        print('| Model')
    else:
        print()

# Save critic, generator and hyperparameters if the IS_max or FID_min is better
save.save_best_AE(test_loss,hyperparameters_AE, checkpoint_path)