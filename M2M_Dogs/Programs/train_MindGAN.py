import torch
from torchvision import transforms, datasets
import torch.optim as optim

import pickle as pkl 
import numpy as np

import time
import os
import shutil

import utils
import models
import evaluate
import save

if not os.path.exists('../checkpoints/Best_Clas_MindGAN/Hyperparameters.txt'):
    print('Veuillez entrainer un classifieur pour MindGAN!!')
    exit()

epochs = 200

data_path, dataset = utils.recup_datas('MindGAN')

if not os.path.exists('./checkpoints/encoder.pth'):
    print('Missing encoder file')
    exit()
    
folder = '../MindGAN'
if not os.path.exists(folder):
    os.makedirs(folder)
    shutil.copyfile('../../Hyperparameters/Hyperparameters_mindgan.csv',  folder + '/Hyperparameters.csv')

# Go to the folder MindGAN
os.chdir(folder)

training_folder = 'Trainings/'
# Folder where trainings are saved
if not os.path.exists(training_folder):
    os.makedirs(training_folder)

os.chdir(training_folder)


# Create the folder by day and time to save the training
folder = time.strftime('%Y_%m_%d_%H_%M_%S')

if not os.path.exists(folder):
    os.makedirs(folder)

print("Toutes les donnees sont enregistrees dans le dossier : " + folder)

# Select hyperparameters for the training
hyperparameters = utils.select_hyperparameters('../Hyperparameters.csv')

# Add the hyparameters at the file Tested_hyperparameters.csv
save.save_tested_hyperparameters(hyperparameters)

# Hyperparameters for the training
[batch_size, num_workers, _, lr, beta1, beta2, gp, epsilon, c_iter, _] = list(hyperparameters.values())

z_size = 128
latent_size = 256

# Folders
data_path = '../../dataset/'
checkpoint_path = folder + '/checkpoints/'
sample_path = folder + '/samples/'

# Make them if they don't exist
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

if not os.path.exists(sample_path):
    os.makedirs(sample_path)

# Save the hyperparameters for the training
save.save_hyperparameters(hyperparameters, folder)

# Transformation  
transform = utils.ToTensor()

Decoder = models.Generator(64,64,z_size=z_size, latent_size = latent_size,mode='AE', nb_channels=3)
state_dict = torch.load('../../checkpoints/decoder.pth')
Decoder.load_state_dict(state_dict)

hyperparameters_AE = {'latent_size' : latent_size}

# Encoding images and save them in folder AE_hyperparameters
filename_encoded_images = utils.encode_images(data_path + '/' + dataset + '/train', hyperparameters_AE)

train_data =  utils.EncodedImages(filename_encoded_images, '.', transform=transform)

train_loader = torch.utils.data.DataLoader( dataset=train_data,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=num_workers,
                                            drop_last=True)

del train_data

nb_classes = len([f for f in os.listdir(data_path + '/' + dataset + '/test') if os.path.isdir(os.path.join(data_path + '/test', f))])

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

C = models.Critic(height, width, latent_size = latent_size, mode='MindGAN', nb_channels=nb_channels)
G = models.Generator(height, width, z_size=z_size, latent_size = latent_size, mode='MindGAN', nb_channels=nb_channels)


# Creation of the classifier which uses to compute the FID and IS
Classifier = models.Pretrain_Classifier(nb_classes)
state_dict = torch.load('../checkpoints/Best_Clas_MindGAN/classifier.pth')
Classifier.load_state_dict(state_dict)
Classifier.eval()


# Trainig on GPU if it's possible
train_on_gpu = torch.cuda.is_available()

if train_on_gpu:
    # move models to GPU
    print('GPU available for training. Models moved to GPU. \n')
    G.cuda()
    C.cuda()
    Decoder.cuda()
else:
    print('Training on CPU.\n')

# Optimizer for Classifier and Generator
c_optimizer = optim.Adam(C.parameters(), lr, [beta1, beta2])
g_optimizer = optim.Adam(G.parameters(), lr, [beta1, beta2])

# Initialisation if IS_max and FID_min
IS_max = - np.inf
FID_min = np.inf
score_IS = -np.inf

IS_mean = 0.
FID_mean = 0.

# Save the time of start
start = time.time()

# Training
for epoch in range(epochs):
    for ii, sample in enumerate(train_loader):
        
        encoded_images = sample['image'].float()
        
        # Reset the gradient
        c_optimizer.zero_grad()

        # Move encoded_images on GPU if we train on GPU
        if train_on_gpu:
            encoded_images = encoded_images.cuda()

        # Computing the critic loss for real images
        c_real_loss = C.expectation_loss(encoded_images)

        # Randomly generation of images
        z = np.random.uniform(-1, 1, size=(batch_size, z_size))
        z = torch.from_numpy(z).float()

        # Move it on GPU if we train on GPU
        if train_on_gpu:
            z = z.cuda()

        # Generation of fake_images by the genrator
        fake_encoded_images = G.forward(z)

        # Computing the critic loss for fake images
        c_fake_loss = C.expectation_loss(fake_encoded_images)

        # Computing gradient penalty and epsilon penalty
        gradient_penalty = C.calculate_gradient_penalty(encoded_images,fake_encoded_images,train_on_gpu)
        epsilon_penalty = C.calculate_epsilon_penalty(encoded_images)

        # Compute the critic loss
        C_loss = - c_real_loss + c_fake_loss + gp * gradient_penalty + epsilon * epsilon_penalty

        # One step in the gradient's descent
        C_loss.backward()
        c_optimizer.step()

        # Computing IS and IS_max, save the model and IS if IS is better
        # Computing FID and FID_max, save the model ans FID if FID is better
        if (ii+1) == len(train_loader):
            real_images = Decoder.forward(encoded_images)
            fake_images = Decoder.forward(fake_encoded_images)
            score_FID, FID_min, affichage = save.save_model_FID(FID_min, real_images, fake_images, Classifier, C, G, checkpoint_path)
            
        # Training of the generator  
        if (ii+1) % c_iter == 0:

            # Reset the gradient
            g_optimizer.zero_grad()

            # Randomly generation of images
            z = np.random.uniform(-1, 1, size=(batch_size, z_size))
            z = torch.from_numpy(z).float()

            # Move it on GPU if we train on GPU
            if train_on_gpu:
                z = z.cuda()

            # Generation of fake_images by the genrator    
            fake_encoded_images = G.forward(z)

            # Compute the generator loss
            G_loss = - C.expectation_loss(fake_encoded_images)

            # One step in the gradient's descent
            G_loss.backward()
            g_optimizer.step()

        # Save log
        if (ii+1) == len(train_loader):
            save.save_log(epoch+1, time.time()-start, C_loss.item(), G_loss.item(),score_IS, score_FID,folder)
            # if epoch >= epochs - 10:
            #     FID_mean += 0.1*score_FID
            #     IS_mean += 0.1*score_IS

    # print discriminator and generator loss
    print('Epoch [{:5d}/{:5d}] \t|\t Time: {:.0f} \t|\t C_loss: {:6.4f} \t|\t G_loss: {:6.4f} \t|\t IS: {:6.4f} \t|\t FID: {:6.4f}'.format(
                epoch+1, epochs, time.time()-start, C_loss.item(), G_loss.item(),score_IS, score_FID), end = "")        
        
    torch.save(C.state_dict(),checkpoint_path + 'critic.pth')
    torch.save(G.state_dict(),checkpoint_path + 'generator.pth')
    print('\t|\t Model saved')

torch.save(C.state_dict(),'../../checkpoints/critic.pth')
torch.save(G.state_dict(),'../../checkpoints/generator.pth')

# Save critic, generator and hyperparameters if the IS_max or FID_min is better
#save.save_best(IS_mean,FID_mean,hyperparameters, checkpoint_path)
