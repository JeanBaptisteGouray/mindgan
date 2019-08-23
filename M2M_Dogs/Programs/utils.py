import random
import numpy as np
import os
import collections
import pandas as pd
from scipy import io
import shutil

from torchvision import transforms, datasets
import torch

import models

def scale(x, features_range=(-1,1)):
    """
        Recale takes in an image x and returns that image, scaled
        with a feature_range of pixel values in features_range. 
    """

    #Rescale x between 0. and 1.
    minix = x.min() 
    maxix = x.max()
    x = (x-minix)/maxix

    #Rescaling to features_range
    mini = np.min(features_range)
    maxi = np.max(features_range)
    length = maxi - mini
    center = length/2.

    return length*x - center

def viz_data_img(images,n=0,labels=None):
    """
        Plot tensor in images, in greyscale
    """
    fig = plt.figure(n,figsize=(25, 4))
    plot_size=images.shape[0]
    for idx in np.arange(plot_size):
        ax = fig.add_subplot(2, plot_size/2, idx+1, xticks=[], yticks=[])
        ax.imshow(images[idx].squeeze(),cmap = 'Greys_r')
        # print out the correct label for each image
        # .item() gets the value contained in a Tensor
        if labels != None:
            ax.set_title(str(labels[idx].item()))


def recup_tested_hyperparameters(folder = '.'):
    """
        Recuperate the hyperparameters already tested
    """
    hyperparameters_tested = []

    if os.path.exists(folder + '/Tested_hyperparameters.csv'):
        with open(folder + '/Tested_hyperparameters.csv','r') as fichier:
            content = fichier.read()

        content = content.split('\n')
        
        if content[-1] == '':
            del content[-1]

        for line in content:
            if line != '':
                line = line.split(';')
                parameters = []
                for parameter in line:
                    if '.' in parameter or 'e' in parameter:
                        parameters.append(float(parameter))
                    else:
                        parameters.append(int(parameter))

                hyperparameters_tested.append(parameters)

    return hyperparameters_tested


def select_hyperparameters(file_hyperparameters, separator = ';'):
    """
        Select randomly hyperparameters in file file_hyperparameters
        Verify if the hyperparamaters are already tested
    """
    with open(file_hyperparameters) as fichier:
        content = fichier.read()

    content = content.split('\n')
    keys = content[0].split(separator)
    del content[0]
    del content[-1]

    hyperparameters_possible = []

    for i in range(len(keys)):
        hyperparameters_possible.append([])

    for line in content:
        line = line.split(separator)
        for i in range(len(line)):
            if line[i] != '':
                if ',' in line[i] or 'e' in line[i] or 'E' in line[i]:
                    hyperparameters_possible[i].append(float(line[i].replace(',','.')))
                else:
                    hyperparameters_possible[i].append(int(line[i]))

    nb_combinaison_possible = 1

    for parameter in hyperparameters_possible:
        nb_combinaison_possible *= len(parameter)

    hyperparameters_tested = recup_tested_hyperparameters()

    hyperparameters = []

    tested = True

    while tested and len(hyperparameters_tested) <= nb_combinaison_possible :
        for parameter in hyperparameters_possible:
            hyperparameters.append(parameter[random.randint(0,len(parameter)-1)])
        tested = hyperparameters in hyperparameters_tested    
        
    return collections.OrderedDict(zip(keys, hyperparameters)) 


def select_hyperparameters_row(file_hyperparameters, separator = ';'):
    """
        Select randomly hyperparameters in file file_hyperparameters
        Verify if the hyperparamaters are already tested
    """
    with open(file_hyperparameters) as fichier:
        content = fichier.read()

    content = content.split('\n')
    keys = content[0].split(separator)
    
    del content[0]
    
    if len(content[-1]) == 0:    
        del content[-1]

    hyperparameters_possible = []

    for line in content:
        line = line.split(separator)
        hyperparameters = []
        
        for i in range(len(line)):
            if line[i] != '':
                if ',' in line[i] or 'e' in line[i] or 'E' in line[i]:
                    hyperparameters.append(float(line[i].replace(',','.')))
                else:
                    hyperparameters.append(int(line[i]))
        hyperparameters_possible.append(hyperparameters)

    nb_hyperparameters_possible = len(content)

    hyperparameters_tested = recup_tested_hyperparameters()

    hyperparameters = []

    tested = True

    while tested and len(hyperparameters_tested) <= nb_hyperparameters_possible :
        hyperparameters = hyperparameters_possible[random.randint(0, len(hyperparameters_possible)-1)]
        tested = hyperparameters in hyperparameters_tested    

    return collections.OrderedDict(zip(keys, hyperparameters)) 


def recup_hyperparameters(file_hyperparameters, separator = ' = '):
    """
        Recuperate the hyperparameters of a NN
    """
    with open(file_hyperparameters) as fichier:
        content = fichier.read()

    content = content.split('\n')
    hyperparameters = []
    keys = []

    for line in content:
        line = line.split(separator)
        keys.append(line[0])
        if '.' in line[1] or 'e' in line[1]:
            hyperparameters.append(float(line[1]))
        else:
            hyperparameters.append(int(line[1]))

    return collections.OrderedDict(zip(keys, hyperparameters))

class EncodedImages(torch.utils.data.Dataset):
    """Encoded Imagess dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.encoded_images = pd.read_csv(csv_file, header = None, sep=';')
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.encoded_images)

    def __getitem__(self, idx):
        image = self.encoded_images.iloc[idx,:-1]
        image = np.ndarray(shape=(len(image),), buffer = np.array(image))
        label = self.encoded_images.iloc[idx, -1]
        label = np.ndarray(shape=(1,), buffer = np.array([label]))
        sample = {'image' : image, 'label' : label}

        return sample


def encode_images(data_path, str_dataset, hyperparameters, transform=transforms.Compose([transforms.ToTensor()]), num_workers=0, pin_memory=False, batch_size = 100):

    folder = '../Encoded_images/AE'
    filename = '/encoded_images.csv'

    for parameter in hyperparameters.values():
        folder += '_' + str(parameter) 

    if not os.path.exists(folder):
        os.makedirs(folder)

    _, train_loader, _, nb_classe = dataset(data_path, str_dataset, batch_size, transform=transform, num_workers=num_workers, pin_memory=pin_memory)

    image = next(iter(train_loader))[0][0]

    nb_channels = image.shape[0]
    height = image.shape[1]
    width = image.shape[2]

    del image

    if len(os.listdir(folder)) == 0:

        print('On encode les images')

        latent_size = hyperparameters['latent_size']

        Encoder = models.Critic(height, width, latent_size = latent_size, mode='AE', nb_channels=nb_channels)
        state_dict = torch.load('../checkpoints/Best_AE/Encoder.pth')
        Encoder.load_state_dict(state_dict)

        train_on_gpu = torch.cuda.is_available()

        if train_on_gpu:
            Encoder.cuda()
        
        for images, labels in train_loader:
            if train_on_gpu:
                images = images.cuda()

            images = scale(images)

            with torch.no_grad():
                encoded_images = Encoder.forward(images)
                
                encoded_images = encoded_images.cpu()

                for i in range(encoded_images.shape[0]):
                    with open(folder + filename, 'a') as fichier:
                        image = encoded_images[i].numpy()
                        for j in range(image.shape[0]):
                            fichier.write(str(image[j]))
                            fichier.write(';')
                        fichier.write(str(labels[i]))
                        fichier.write('\n')

        print('Fin de l\'encodage des images')

    return folder + filename, nb_classe, height, width, nb_channels

def recup_scores(score, folder, bigger_is_better=False, nb_values = 10):

    vect = np.ones(nb_values)/nb_values
    
    hyperparamaters = recup_hyperparameters(folder + '/Hyperparameters.txt')
    epochs = hyperparamaters['epochs']

    if not os.path.exists(folder + '/log.csv'):
        if bigger_is_better:
            return 0
        else:
            return np.inf
    else:
        datas = pd.read_csv(folder + '/log.csv', sep=';')
        if len(datas) == epochs:
            Score_mean = np.dot(datas[score].iloc[-nb_values:].values, vect)
        else:
            if bigger_is_better:
                return 0
            else:
                return np.inf

    return Score_mean

def dataset(data_path, dataset, batch_size, transform=transforms.Compose([transforms.ToTensor()]), num_workers=0, pin_memory=False):
    
    possible_datasets = ('MNIST', 'FashionMNIST', 'KMNIST', 'SVHN', 'CIFAR10', 'CIFAR100', 'STL10', 'LSUN', 'ImageNet', 'Cat_Dog', 'Doggos_data', 'Dog_Breed')

    if dataset in possible_datasets:

        data_path = data_path + '/' + dataset

        if dataset == 'MNIST':

            train_data = datasets.MNIST(root = data_path,train=True,transform=transform,download=True)
            test_data = datasets.MNIST(root = data_path,train=False,transform=transform,download=True)

            train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=num_workers,
                                                    pin_memory=pin_memory)

            test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        num_workers=num_workers,
                                                        pin_memory=pin_memory)

            nb_classe = 10

        elif dataset == 'FashionMNIST':

            train_data = datasets.FashionMNIST(root=data_path, train=True, transform=transform, download=True)
            test_data = datasets.FashionMNIST(root=data_path, train=False, transform=transform, download=True)

            train_loader = torch.utils.data.DataLoader( dataset=train_data,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        num_workers=num_workers,
                                                        pin_memory=pin_memory)

            test_loader = torch.utils.data.DataLoader(  dataset=test_data,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        num_workers=num_workers,
                                                        pin_memory=pin_memory)
            nb_classe =10

        elif dataset == 'KMNIST':

            train_data = datasets.KMNIST(root=data_path, train=True, transform=transform, download=True)
            test_data = datasets.KMNIST(root=data_path, train=False, transform=transform, download=True)

            train_loader = torch.utils.data.DataLoader( dataset=train_data,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        num_workers=num_workers,
                                                        pin_memory=pin_memory)

            test_loader = torch.utils.data.DataLoader(  dataset=test_data,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        num_workers=num_workers,
                                                        pin_memory=pin_memory) 
            nb_classe = 10

        elif dataset == 'SVHN':
            
            train_data = datasets.SVHN(root=data_path, split='train',transform=transform,download=True)
            test_data = datasets.SVHN(root=data_path, split='test',transform=transform,download=True)

            train_loader = torch.utils.data.DataLoader( dataset=train_data,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        num_workers=num_workers,
                                                        pin_memory=pin_memory)

            test_loader = torch.utils.data.DataLoader(  dataset=test_data,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        num_workers=num_workers,
                                                        pin_memory=pin_memory)
            
            nb_classe = 10

        elif dataset == 'CIFAR10':
            
            train_data = datasets.CIFAR10(root=data_path, train=True, transform=transform, download=True)
            test_data = datasets.CIFAR10(root=data_path, train=False, transform=transform, download=True)

            train_loader = torch.utils.data.DataLoader( dataset=train_data,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        num_workers=num_workers,
                                                        pin_memory=pin_memory)

            test_loader = torch.utils.data.DataLoader(  dataset=test_data,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        num_workers=num_workers,
                                                        pin_memory=pin_memory)
        
            nb_classe = 10

        elif dataset == 'CIFAR100':

            train_data = datasets.CIFAR100(root=data_path, train=True, transform=transform, download=True)
            test_data = datasets.CIFAR100(root=data_path, train=False, transform=transform, download=True)

            train_loader = torch.utils.data.DataLoader( dataset=train_data,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        num_workers=num_workers,
                                                        pin_memory=pin_memory)

            test_loader = torch.utils.data.DataLoader(  dataset=test_data,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        num_workers=num_workers,
                                                        pin_memory=pin_memory)

            nb_classe = 100

        elif dataset == 'STL10':

            train_data = datasets.STL10(root=data_path, split='train', transform=transform, download=True)
            test_data = datasets.STL10(root=data_path, split='test', transform=transform, download=True)

            train_loader = torch.utils.data.DataLoader( dataset=train_data,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        num_workers=num_workers,
                                                        pin_memory=pin_memory)

            test_loader = torch.utils.data.DataLoader(  dataset=test_data,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        num_workers=num_workers,
                                                        pin_memory=pin_memory)

            nb_classe = 10

        elif dataset == 'LSUN':

            train_data = datasets.LSUN(data_path + '/' + dataset, classes='train', transform=transform)
            test_data = datasets.LSUN(data_path + '/' + dataset, classes='test', transform=transform)

            train_loader = torch.utils.data.DataLoader( dataset=train_data,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        num_workers=num_workers,
                                                        pin_memory=pin_memory)

            test_loader = torch.utils.data.DataLoader(  dataset=test_data,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        num_workers=num_workers,
                                                        pin_memory=pin_memory)

            nb_classe = 10

        elif dataset == 'ImageNet':

            train_data = datasets.ImageNet(root=data_path, split='train', download=True)
            test_data = datasets.ImageNet(root=data_path, split='test', download=True)

            train_loader = torch.utils.data.DataLoader( dataset=train_data,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        num_workers=num_workers,
                                                        pin_memory=pin_memory)

            test_loader = torch.utils.data.DataLoader(  dataset=test_data,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        num_workers=num_workers,
                                                        pin_memory=pin_memory)

            nb_classe = 1000

        elif dataset == 'Cat_Dog' or dataset == 'Doggos_data' :
            chemin = data_path.split('/')
            del chemin[-1]
            data_path = "/".join(chemin)
            if not os.path.exists(data_path + '/Cat_Dog/train') or not os.path.exists(data_path + '/Cat_Dog/test'):
                if not os.path.exists(data_path + '/Cat_Dog_data.zip'):
                    os.system('wget -cP ' + data_path + ' https://s3.amazonaws.com/content.udacity-data.com/nd089/Cat_Dog_data.zip')
                print('Decompression des donnees')
                os.system("unzip -q " + data_path + '/Cat_Dog_data.zip -d ' + data_path)
                if os.path.exists(data_path + '/__MACOSX'):
                    os.system('rm -r ' + data_path + '/__MACOSX')
                os.remove(data_path + '/Cat_Dog_data.zip')
                os.rename(data_path + '/Cat_Dog_data', data_path + '/Cat_Dog')
                shutil.copytree(data_path + '/Cat_Dog/train/dog', data_path + '/Doggos_data/train/dog')
                shutil.copytree(data_path + '/Cat_Dog/test/dog', data_path + '/Doggos_data/test/dog')

            data_path += '/' + dataset
            
            train_data = datasets.ImageFolder(data_path + '/train', transform=transform)
            test_data = datasets.ImageFolder(data_path + '/test', transform=transform)

            train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

            nb_classe = 2

        elif dataset == 'Dog_Breed':

            if not os.path.exists(data_path):
                os.makedirs(data_path)
            
            if not os.path.exists(data_path + '/train') or not os.path.exists(data_path + '/test'):
                if not os.path.exists(data_path + '/lists'):
                    # Téléchargement et extraction des listes pour le train et test
                    if not os.path.exists(data_path + '/lists//lists.tar'):
                        os.system('wget -cP' + data_path + '/lists http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar')
                    os.system("tar xvf " + data_path + '/lists/lists.tar -C ' + data_path + '/lists')
                    os.remove(data_path + '/lists/lists.tar')
                    print()

                if not os.path.exists(data_path + '/Images'):
                    # Téléchargement et extraction des images
                    if not os.path.exists(data_path + '/images.tar'):
                        os.system('wget -cP ' + data_path + ' http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar')
                    print('Decompression des images')
                    os.system("tar xf " + data_path + '/images.tar -C ' + data_path )
                    os.remove(data_path + '/images.tar')
            
            # Création des dossiers train et test
            train_dir = data_path + '/train/'
            test_dir = data_path + '/test/'

            if not os.path.exists(train_dir):
                os.makedirs(train_dir)

            if len(os.listdir(train_dir)) != 120:
                train_filenames = io.loadmat(data_path + '/lists/train_list.mat')['annotation_list']

                print('Creation du dataset train')
                for i in range(train_filenames.shape[0]) :
                    print('\r{}/{}'.format(i+1, train_filenames.shape[0]), end='')
                    filename = train_filenames[i][0][0]
                    dirname = filename.split('/')[0]

                    if not os.path.exists(train_dir + dirname):
                        os.makedirs(train_dir + dirname)
                    if not os.path.exists(train_dir + filename + '.jpg') :
                        shutil.copyfile(data_path + '/Images/' + filename + '.jpg', train_dir + filename + '.jpg')
                print()

            if not os.path.exists(test_dir) :
                os.makedirs(test_dir)
            
            if len(os.listdir(test_dir)) != 120:
                test_filenames = io.loadmat(data_path + '/lists/test_list.mat')['annotation_list']

                print('Creation du dataset test')
                for i in range(test_filenames.shape[0]) :
                    print('\r{}/{}'.format(i+1, test_filenames.shape[0]), end='')
                    filename = test_filenames[i][0][0]
                    dirname = filename.split('/')[0]

                    if not os.path.exists(test_dir + dirname):
                        os.makedirs(test_dir + dirname)

                    if not os.path.exists(test_dir + filename + '.jpg') :
                        shutil.copyfile(data_path + '/Images/' + filename + '.jpg', test_dir + filename + '.jpg')
                print()
            
            train_data = datasets.ImageFolder(train_dir, transform=transform)
            test_data = datasets.ImageFolder(test_dir, transform=transform)

            train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

            nb_classe = 120
        
        return data_path, train_loader, test_loader, nb_classe 

    else :
        print('Veuillez entrer un dataset parmi les propositions suivantes:')
        
        for f in possible_datasets :
            print(f)
    
        exit()

def recup_datas(key, filename='datas.txt'):
    
    with open(filename, 'r') as fichier:
        content = fichier.read()
    
    content = content.split('\n')
    
    datasets = []
    keys = []
    
    for line in content:
        if line != '' and '#' not in line:
            line = line.split(' = ')
            keys.append(line[0])
            datasets.append(line[1])
    
    list_dataset = collections.OrderedDict(zip(keys, datasets))
    
    return list_dataset['path'], list_dataset[key]