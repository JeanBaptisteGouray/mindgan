import torch
import torch.nn as nn

import numpy as np
from time import time
import json
import os

from torchvision.utils import save_image

class TrainerWGAN():
    def __init__(self, generator, critic, hyperparameters, config, optims, decoder=None, classifier=None):
        self.generator = generator
        self.critic = critic
        self.hyperparameters = hyperparameters
        self.config = config
        self.optim_generator = optims[0]
        self.optim_critic = optims[1]
        self.classifier = classifier
        self.decoder = decoder

    def loop(self, data_loader):
        if not os.path.exists(self.config['out_dir'] + '/'+ self.config['name']):
            os.makedirs(self.config['out_dir'] +'/'+ self.config['name'])
        with open(self.config['out_dir'] +'/'+ self.config['name'] + '/parameters.json', 'w') as f:
            json.dump(self.hyperparameters, f, indent=2)

        critic_loss_list = list()
        generator_loss_list = list()
        time_list = list()

        print('Training ' + self.config['name'] + ' begins !')
        start = time()

        for epoch in range(self.hyperparameters['epochs']):
            Critic_loss = 0
            Generator_loss = 0
            for data in data_loader:
                if type(data) == list:
                    data = data[0]
                elif type(data) == dict:
                    data = data['latent']

                if self.config['train_on_gpu']:
                    data = data.cuda()

                bs = data.shape[0]
                for _ in range(self.hyperparameters['c_iter']):
                    self.optim_critic.zero_grad()
                    z = torch.from_numpy(
                        np.random.uniform(-1, 1, size=(bs, self.hyperparameters['z_dim']))).float()

                    if self.config['train_on_gpu']:
                        z = z.cuda()

                    fake_data = self.generator.forward(z)

                    critic_losses_batch = self.calculate_critic_losses(
                        fake_data, data)
                    total_critic_loss = self.calculate_total_critic_loss(
                        critic_losses_batch)
                    Critic_loss += total_critic_loss.item() / (len(data_loader) *
                                                               self.hyperparameters['c_iter'])

                    total_critic_loss.backward()
                    self.optim_critic.step()

                self.optim_generator.zero_grad()
                z = torch.from_numpy(
                    np.random.uniform(-1, 1, size=(bs, self.hyperparameters['z_dim']))).float()

                if self.config['train_on_gpu']:
                    z = z.cuda()

                fake_data = self.generator(z)

                generator_losses_batch = self.calculate_generator_losses(
                    fake_data)
                total_generator_loss = self.calculate_total_generator_loss(
                    generator_losses_batch)
                Generator_loss += total_generator_loss.item() / len(data_loader)

                total_generator_loss.backward()
                self.optim_generator.step()

            end = time()
            time_list.append(end-start)
            critic_loss_list.append(Critic_loss)
            generator_loss_list.append(Generator_loss)


            self.logs(epoch, time_list, critic_loss_list,
                        generator_loss_list)

        print('Training is over ! Time : {:.5f} secs ({:.5f} secs per epoch)'.format(
            end-start, (end-start)/self.hyperparameters['epochs']))

    def calculate_critic_losses(self, fake, real):
        c_fake_loss = self.critic.expectation_loss(fake)
        c_real_loss = self.critic.expectation_loss(real)
        gradient_penalty = self.critic.calculate_gradient_penalty(
            real, fake, self.config['train_on_gpu'])
        epsilon_penalty = self.critic.calculate_epsilon_penalty(real)
        return c_real_loss, c_fake_loss, gradient_penalty, epsilon_penalty

    def calculate_generator_losses(self, fake):
        c_fake_loss = self.critic.expectation_loss(fake)
        return -c_fake_loss

    def calculate_total_critic_loss(self, losses):
        c_real_loss, c_fake_loss, gradient_penalty, epsilon_penaly = losses
        return - c_real_loss + c_fake_loss + self.hyperparameters['gp'] * gradient_penalty + self.hyperparameters['epsilon']*epsilon_penaly

    def calculate_total_generator_loss(self, losses):
        return losses

    def logs(self, epoch, time_list, critic_loss_list, generator_loss_list):
        print('Epoch : {}/{} | Critic loss : {:.5f} | Generator loss : {:.5f}'.format(epoch+1, self.hyperparameters['epochs'],
                                                                                        critic_loss_list[-1],
                                                                                        generator_loss_list[-1]), end='')

        if self.config['train_on_gpu']:
            self.critic.cpu()
            self.generator.cpu()

        state_dict_c = self.critic.state_dict()
        state_dict_g = self.generator.state_dict()

        if self.config['train_on_gpu']:
            self.critic.cuda()
            self.generator.cuda()

        print(' | Models saved !')

        save_dict = {
            'Time': np.array(time_list),
            'Critic': state_dict_c,
            'Generator': state_dict_g,
            'Critic loss': critic_loss_list,
            'Generator loss': generator_loss_list,
            'optims': [self.optim_generator.state_dict(), self.optim_critic.state_dict()]
        }

        torch.save(save_dict, self.config['out_dir'] + '/' + self.config['name'] + '/model_'+ str(epoch) + '.pth')
