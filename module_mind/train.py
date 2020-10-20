import torch
import torch.optim as optim
from torchvision import datasets, transforms

import os

import module_mind.models as models
from module_mind.trainer import TrainerWGAN
from module_mind.data_loader import EncodedFFHQ


def train_mgan_ffhq(hyperparameters, config):
    if not os.path.exists(config['out_dir']):
        os.makedirs(config['out_dir'])

    Critic = models.FFHQCritic(hyperparameters['hidden_critic_c'])
    Generator = models.FFHQGenerator(hyperparameters['z_dim'], hyperparameters['hidden_generator_c'])

    if config['train_on_gpu']:
        Critic.cuda()
        Generator.cuda()
        print('Using GPU')
    else:
        print('Using CPU')

    optimizer_g = optim.Adam(Generator.parameters(), hyperparameters['lr'], betas=hyperparameters['betas'])
    optimizer_c = optim.Adam(Critic.parameters(), hyperparameters['lr'], betas=hyperparameters['betas'])

    trainer = TrainerWGAN(Generator, Critic, hyperparameters,
                          config, [optimizer_g, optimizer_c])

    train_data = EncodedFFHQ(data_path=config['data_path'])

    data_loader = torch.utils.data.DataLoader(
        dataset=train_data, batch_size=hyperparameters['batch_size'], shuffle=True, num_workers=config['num_workers'], pin_memory=config['pin_memory'])

    trainer.loop(data_loader)


if __name__ == '__main__':
    hyperparameters_ffhq = {
        'hidden_critic_c': [512, 256, 128],
        'hidden_generator_c': [128, 256, 512],
        'epochs': 300,
        'lr': 1e-3,
        'betas': [.0, .5],
        'batch_size': 128,
        'z_dim': 128,
        'gp': 10.,
        'epsilon': 1e-2,
        'c_iter': 5
    }

    config_ffhq = {
        'name' : 'celeba_from_ffhq',
        'num_workers': 32,
        'pin_memory': True,
        'data_path': 'module_mind/Dataset/Encoded_from_FFHQ',
        'out_dir': 'module_mind/output',
        'train_on_gpu': True
    }

    train_mgan_ffhq(hyperparameters_ffhq, config_ffhq)
