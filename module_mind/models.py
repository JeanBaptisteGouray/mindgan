import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import autograd

class FFHQCritic(nn.Module):
    def __init__(self,  dense):
        super().__init__()
        dense_layers = list()
        if len(dense) != 0:
            dense_layers = [nn.Linear(512, dense[0]), nn.ReLU()]
            for i in range(len(dense)-1):
                dense_layers.append(nn.Linear(dense[i], dense[i+1]))
                dense_layers.append(nn.ReLU())
            dense_layers.append(nn.Linear(dense[-1], 1))
        else:
            dense_layers = [nn.Linear(512, 1)]

        self.model = nn.Sequential(*dense_layers)
        self.parameters = self.model.parameters

    def forward(self, latent):
        if len(latent.shape) >= 3:
            latent = latent.view(latent.shape[0],512)
        return self.model(latent)

    def expectation_loss(self, img):
        return self.forward(img).mean(0).view(1)

    def calculate_gradient_penalty(self, real_images, fake_images, train_on_gpu):
        batch_size = real_images.shape[0]
        
        t = torch.FloatTensor(batch_size, 1).uniform_(0, 1)
        t = t.expand(batch_size, 512)

        if train_on_gpu:
            t = t.cuda()

        x_t = t * real_images + ((1 - t) * fake_images.view(batch_size,  512))

        if train_on_gpu:
            x_t = x_t.cuda()

        x_t = Variable(x_t, requires_grad=True)

        c_out = self.forward(x_t)

        gradients = autograd.grad(outputs=c_out, inputs=x_t,
                                  grad_outputs=torch.ones(
                                      c_out.size()).cuda() if train_on_gpu else torch.ones(
                                      c_out.size()),
                                  create_graph=True, retain_graph=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def calculate_epsilon_penalty(self, real_images):
        d_out = self.forward(real_images)
        d_out_squared = torch.pow(d_out, 2)
        return d_out_squared.mean()

class FFHQGenerator(nn.Module):
    def __init__(self, z_dim, dense):
        super().__init__()
        dense_layers = list()
        if len(dense) != 0:
            dense_layers = [nn.Linear(z_dim, dense[0]), nn.ReLU()]
            for i in range(len(dense)-1):
                dense_layers.append(nn.Linear(dense[i], dense[i+1]))
                dense_layers.append(nn.ReLU())
            dense_layers.append(nn.Linear(dense[-1], 512))
        else:
            dense_layers = [nn.Linear(z_dim, 512)]

        self.model = nn.Sequential(*dense_layers)
        self.parameters = self.model.parameters

    def forward(self, z):
        return self.model(z)

    def sample(self, z):
        return self.forward(z).view(z.shape[0],1,512).repeat(1,18,1)