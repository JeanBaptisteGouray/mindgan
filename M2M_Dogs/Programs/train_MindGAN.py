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

if not os.path.exists('../checkpoints/Best_AE/Encoder.pth'):
    print('Missing encoder file')
    exit()
    
epochs = 200

data_path, dataset = utils.recup_datas('MindGAN')

print('Les datasets se trouvent a l\'emplacement :', data_path)
print('Le dataset utilise est :', dataset)

folder = '../MindGAN'

if not os.path.exists(folder):
    os.makedirs(folder)

if not os.path.exists(folder + '/Hyperparameters.csv'):
    shutil.copyfile('../../Hyperparameters/Hyperparameters_mindgan.csv',  folder + '/Hyperparameters.csv')

# Go to the folder MindGAN
os.chdir(folder)

folder = 'Trainings/'

# Create the folder by day and time to save the training
folder += time.strftime('%Y_%m_%d_%H_%M_%S')

if not os.path.exists(folder):
    os.makedirs(folder)

print("Toutes les donnees sont enregistrees dans le dossier : " + folder)

# Select hyperparameters for the training
hyperparameters = utils.select_hyperparameters('./Hyperparameters.csv')
print('Hyperparameters = ', hyperparameters)

# Add the hyparameters at the file Tested_hyperparameters.csv
save.save_tested_hyperparameters(hyperparameters)

# Hyperparameters for the training
[batch_size, num_workers, _, lr, beta1, beta2, gp, epsilon, c_iter, _] = list(hyperparameters.values())

z_size = 128
hyperparameters_AE = utils.recup_hyperparameters('../checkpoints/Best_AE/Hyperparameters.txt')
latent_size = hyperparameters_AE['latent_size']

# Folders
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
transform = transforms.Compose([transforms.Resize(140),
                                transforms.CenterCrop(128),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
                                ])

# Encoding images and save them in folder AE_hyperparameters
filename_encoded_images, height, width, nb_channels = utils.encode_images(data_path, dataset, hyperparameters_AE, transform=transform)

train_data =  utils.EncodedImages(filename_encoded_images, '.', transform=transform)

train_loader = torch.utils.data.DataLoader( dataset=train_data,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=num_workers,
                                            drop_last=True)
del train_data

nb_classes = len([f for f in os.listdir(data_path + '/' + dataset + '/test') if os.path.isdir(os.path.join(data_path + '/' + dataset + '/test', f))])

print('Il y a {} classes'.format(nb_classes))
print('La taille des images est de : ({},{},{})'.format(nb_channels, height, width))

# Parameter for the print
print_every = len(train_loader)//1

# Recuperation Decoder
Decoder = models.Generator(height, width, z_size=z_size, latent_size = latent_size,mode='AE', nb_channels=nb_channels)
state_dict = torch.load('../checkpoints/Best_AE/Decoder.pth')
Decoder.load_state_dict(state_dict)

# Creation of the crtic and the generator
C = models.Critic(height, width, latent_size = latent_size, mode='MindGAN', nb_channels=nb_channels)
G = models.Generator(height, width, z_size=z_size, latent_size = latent_size, mode='MindGAN', nb_channels=nb_channels)


# Creation of the classifier which uses to compute the FID and IS
Classifier = models.MLP(nb_classes)
state_dict = torch.load('../checkpoints/Best_Clas_MindGAN/classifier.pth')
Classifier.load_state_dict(state_dict)
Classifier.eval()


# Trainig on GPU if it's possible
train_on_gpu = torch.cuda.is_available()

if train_on_gpu:
    # move models to GPU
    G.cuda()
    C.cuda()
    Decoder.cuda()
    Classifier.cuda()
    print('GPU available for training. Models moved to GPU. \n')
else:
    print('Training on CPU.\n')

# Optimizer for Classifier and Generator
c_optimizer = optim.Adam(C.parameters(), lr, [beta1, beta2])
g_optimizer = optim.Adam(G.parameters(), lr, [beta1, beta2])

# Initialisation of test_loss_min
FID_min = np.inf
IS_max = -np.inf

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
            score_IS = evaluate.inception_score(fake_images,Classifier)
            score_FID = evaluate.fid(real_images, fake_images, Classifier)
        
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
    save.save_log(epoch+1, time.time()-start, C_loss.item(), G_loss.item(),score_IS, score_FID,folder)

    # print discriminator and generator loss
    print('Epoch [{:5d}/{:5d}] | Time: {:.0f} | C_loss: {:6.4f} | G_loss: {:6.4f} | IS: {:6.4f} | FID: {:6.4f} '.format(
                epoch+1, epochs, time.time()-start, C_loss.item(), G_loss.item(),score_IS, score_FID), end = "")        
    
    IS_max, FID_min, affichage = save.save_model_IS_FID(score_IS, score_FID, IS_max, FID_min, C, G, checkpoint_path)
    
    if affichage:
        print('| Model saved')
    else:
        print()

# Save critic, generator and hyperparameters if the IS_max or FID_min is better
save.save_best_IS_FID(IS_max, FID_min, hyperparameters, checkpoint_path)
