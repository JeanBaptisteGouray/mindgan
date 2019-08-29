import torch
from torchvision import transforms, datasets
import torch.optim as optim
import torch.nn as nn

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

seed = 56356274

torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed) 

# torch.backends.cudnn.benchmark = True

epochs = 200
num_workers = 32
pin_memory = True

data_path, dataset = utils.recup_datas('MindGAN')

print('Les datasets se trouvent a l\'emplacement :', data_path)
print('Le dataset utilise est :', dataset)

folder = '../VAE'

if not os.path.exists(folder):
    os.makedirs(folder)

os.chdir(folder)

folder = 'Trainings/'

# Create the folder by day and time to save the training
folder += time.strftime('%Y_%m_%d_%H_%M_%S')

if not os.path.exists(folder):
    os.makedirs(folder)

print("Toutes les donnees sont enregistrees dans le dossier : " + folder)


batch_size, latent_size, conv_dim, lr = 128, 20, 2, 0.001


checkpoint_path = folder + '/checkpoints/'

# Make them if they don't exist
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

# Transformation  
transform = transforms.Compose([transforms.Resize(140),
                                transforms.CenterCrop(128),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
                                ])

data_path, train_loader, test_loader, nb_classes = utils.dataset(data_path, dataset, batch_size, transform, num_workers=num_workers, pin_memory=pin_memory)

image = next(iter(train_loader))[0][0]

nb_channels = image.shape[0]
height = image.shape[1]
width = image.shape[2]

del image

print('Il y a {} classes'.format(nb_classes))
print('La taille des images est de : ({},{},{})'.format(nb_channels, height, width))

# Parameter for the print
print_every = len(train_loader)//1


VAE = models.VAE(height, width, conv_dim = conv_dim, latent_size = latent_size, nb_channels=nb_channels)

# Creation of the classifier which uses to compute the FID and IS
Classifier = models.MLP(nb_classes)
state_dict = torch.load('../checkpoints/Best_Clas_MindGAN/classifier.pth')
Classifier.load_state_dict(state_dict)
Classifier.eval()

train_on_gpu = torch.cuda.is_available()

if train_on_gpu:
    # move models to GPU
    VAE.cuda()
    print('GPU available for training. Models moved to GPU. \n')
else:
    print('Training on CPU.\n')

reconstruction_function = nn.MSELoss(size_average=False)


def loss_function(recon_x, x, mu, logvar):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    BCE = reconstruction_function(recon_x, x)  # mse loss
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return BCE + KLD

optimizer = optim.Adam(VAE.parameters(), lr=lr)

test_loss_min = np.inf

start = time.time()
for epoch in range(epochs):
    train_loss = 0.
    score_IS = 0.
    score_FID = 0.
    for real_images, _ in train_loader:
        # Move real_images on GPU if we train on GPU
        if train_on_gpu:
            real_images = real_images.cuda()

        optimizer.zero_grad()

        recon_batch, mu, logvar = VAE(real_images)
        loss = loss_function(recon_batch, real_images, mu, logvar)
        loss.backward()
        train_loss += loss
        optimizer.step()
        

########## WARNING : Calcul de l'IS et du FID avec des images reconstruites ici. A changer pour de la "vraie" génération ################
        score_IS += evaluate.inception_score(recon_batch,Classifier)
        score_FID += evaluate.fid(real_images, recon_batch, Classifier)

    test_loss = 0.
    with torch.no_grad():
        VAE.eval()
        for real_images, _ in test_loader:

            # Move images on GPU if we train on GPU
            if train_on_gpu:
                real_images = real_images.cuda()

            recon_batch, mu, logvar = VAE(real_images)
            loss = loss_function(recon_batch, real_images, mu, logvar)
            test_loss += loss
        VAE.train()
    
    if test_loss_min > test_loss.item()/len(test_loader):
        test_loss_min = test_loss.item()/len(test_loader)
        torch.save(VAE.state_dict(), checkpoint_path + 'VAE.pth')
        affichage = True
    else:
        affichage = False

    save.save_log_VAE(epoch+1, time.time()-start, train_loss.item()/len(train_loader), test_loss.item()/len(test_loader),score_IS/len(train_loader), score_FID/len(train_loader),folder)

    print('Epoch [{:5d}/{:5d}] | Time: {:.0f} | Training loss: {:6.4f} | Testing loss: {:6.4f} | FID: {:6.4f} | IS: {:6.4f}'.format(
        epoch+1, epochs, time.time()-start, train_loss.item()/len(train_loader), test_loss.item()/len(test_loader),score_FID/len(train_loader),score_IS/len(train_loader)), end=' ')
    
    
    if affichage:
        print('| Model saved')
    else:
        print()




