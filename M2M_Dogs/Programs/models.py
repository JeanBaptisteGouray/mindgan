import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
from torchvision import models as pretrain
from torch import autograd

class Critic(nn.Module):
    def __init__(self, height, width, conv_dim = 32, latent_size = 128,mode = None, nb_channels=1):
        super(Critic, self).__init__()
        self.mode = mode

        if mode != 'MindGAN': 
            self.conv1 = conv(nb_channels,conv_dim,4,2,1)       # height/2, width/2
            self.conv2 = conv(conv_dim,2*conv_dim,4,2,1)        # height/4, width/4
            self.conv3 = conv(2*conv_dim,4*conv_dim,4,2,1)      # height/8, width/8
            self.conv4 = conv(4*conv_dim,8*conv_dim,4,2,1)      # height/16, width/16
            self.fc1 = nn.Linear(int(height/16)*int(width/16)*8*conv_dim,latent_size)    

        if mode != 'AE':
            self.fc2 = nn.Linear(latent_size,conv_dim)
            self.fc3 = nn.Linear(conv_dim,1)

    def forward(self, x):

        if self.mode != 'MindGAN':
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = F.relu(self.conv4(x))
            x = x.view(x.shape[0],-1)
            x = F.relu(self.fc1(x))

            if self.mode == 'AE':
                return x
        
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def expectation_loss(self, x): 
        return self.forward(x).mean(0).view(1)

    def calculate_gradient_penalty(self, real_images, fake_images,train_on_gpu):
        batch_size = real_images.shape[0]

        if self.mode != 'MindGAN':
            t = torch.FloatTensor(batch_size,1,1,1).uniform_(0,1)
            t = t.expand(batch_size, real_images.size(1), real_images.size(2), real_images.size(3))
        else:
            t = torch.FloatTensor(batch_size,1).uniform_(0,1)
            t = t.expand(batch_size, real_images.size(1))
        
        if train_on_gpu:
            t = t.cuda()
        else:
            t = t
        x_t = t * real_images + ((1 - t) * fake_images)

        if train_on_gpu:
            x_t = x_t.cuda()
        else:
            x_t = x_t

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
        d_out_squared = torch.pow(d_out,2)
        return d_out_squared.mean()

class Generator(nn.Module):
    def __init__(self, height, width, z_size, conv_dim = 32, latent_size = 128, mode = None, nb_channels=1):
        super(Generator, self).__init__()
        self.conv_dim = conv_dim
        self.mode = mode
        self.height = height
        self.width = width

        if self.mode != 'AE':
            self.fc1 = nn.Linear(z_size,conv_dim)
            self.fc2 = nn.Linear(conv_dim,latent_size)

        if self.mode != 'MindGAN':
            self.fc3 = nn.Linear(latent_size,int(self.height/16)*int(self.width/16)*8*conv_dim)
            self.Tconv1 = Tconv(8*conv_dim,4*conv_dim,2,2,0)                            # height/8, widht/8
            self.Tconv2 = Tconv(4*conv_dim,4*conv_dim,2,2,0)                              # height/2, width/2
            self.Tconv3 = Tconv(4*conv_dim,2*conv_dim,2,2,0)
            self.Tconv4 = Tconv(2*conv_dim,nb_channels,2,2,0,batch_norm=False)            # height, width
            self.tanh = nn.Tanh()

    def forward(self, x):

        if self.mode  != 'AE':
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))

            if self.mode == 'MindGAN':
                return x

        x = F.relu(self.fc3(x))
        x = x.view(x.shape[0], 8*self.conv_dim, int(self.height/16), int(self.width/16))
        x = F.relu(self.Tconv1(x))
        x = F.relu(self.Tconv2(x))
        x = F.relu(self.Tconv3(x))
        x = self.tanh(self.Tconv4(x))
        return x


class Classifier(nn.Module):
    def __init__(self, height, width, out_dim, nb_channels=1, conv_dim = 32, p=0.00125):
        super(Classifier, self).__init__()
        self.conv1 = conv(nb_channels,conv_dim,3,1,1)      # dim : H/2, W/2
        self.conv2 = conv(conv_dim,2*conv_dim,3,1,1)       # dim : H/4, W/4
        self.conv3 = conv(2*conv_dim,4*conv_dim,3,1,1)     # dim : H/8, W/8
        self.fc = nn.Linear(int(height/8)*int(width/8)*4*conv_dim,out_dim)
        self.pool = nn.MaxPool2d(2,2)
        self.drop = nn.Dropout(p=p)

    def forward(self, x,get_activations = False):
        x = self.drop(self.pool(F.relu(self.conv1(x))))
        x = self.drop(self.pool(F.relu(self.conv2(x))))    
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.shape[0], -1)  

        if not get_activations:
            x = F.log_softmax(self.fc(x), dim = 1)
            
        return x

class MLP(nn.Module):
    def __init__(self,out_dim, hidden_units = [1024],transfer = 'densenet121', nb_channels=1, conv_dim = 32, p=0.00125,height = 28, width = 28):
        super(MLP, self).__init__()
        self.transfer = transfer
        if transfer is not None:
            if transfer == 'densenet121':
                self.model = pretrain.densenet121(pretrained=True)
                in_dim = 1024

            if transfer == 'resnet101':
                self.model = pretrain.densenet121(pretrained=True)
                in_dim = 2048

            if transfer == 'vgg19':
                self.model = pretrain.vgg19(pretrained=True)
                in_dim = 25088

            for param in self.model.parameters():
                param.requires_grad = False
            if len(hidden_units)==0:
                layers = [nn.Linear(in_dim,out_dim)]
            else:
                layers = [nn.Linear(in_dim,hidden_units[0]),nn.ReLU()]
                if len(hidden_units) > 1:
                    for i in range(len(hidden_units)-1):
                        layers.append(nn.Linear(hidden_units[i],hidden_units[i+1]))
                        layers.append(nn.ReLU())
                layers.append(nn.Linear(hidden_units[-1],out_dim))

            self.sequential = nn.Sequential(*layers)
            self.model.classifier = self.sequential
            self.parameters = self.model.classifier.parameters
        else:
            
            self.conv1 = conv(nb_channels,conv_dim,3,1,1)      # dim : H/2, W/2
            self.conv2 = conv(conv_dim,2*conv_dim,3,1,1)       # dim : H/4, W/4
            self.conv3 = conv(2*conv_dim,4*conv_dim,3,1,1)     # dim : H/8, W/8
            self.fc = nn.Linear(int(height/8)*int(width/8)*4*conv_dim,out_dim)
            self.pool = nn.MaxPool2d(2,2)
            self.drop = nn.Dropout(p=p)

            layers = [self.conv1,nn.ReLU(),self.pool,self.drop,self.conv2,nn.ReLU(),self.pool,self.drop,self.conv3,nn.ReLU(),self.pool,Flatten(),self.fc]
            self.sequential = nn.Sequential(*layers)
        

    def forward(self, x,get_activations = False):
        if self.transfer is not None:
            x = self.model.forward(x)
        else:
            x = self.sequential(x)
        if not get_activations:
            x = F.log_softmax(x, dim = 1)
            
        return x

class VAE(nn.Module):
    def __init__(self, height, width, conv_dim = 32, latent_size = 20, nb_channels=1):
        super(VAE, self).__init__()
        self.conv_dim = conv_dim
        self.height = height
        self.width = width
        #Encode
        self.conv1 = conv(nb_channels,conv_dim,4,2,1)       # height/2, width/2
        self.conv2 = conv(conv_dim,2*conv_dim,4,2,1)        # height/4, width/4
        self.conv3 = conv(2*conv_dim,4*conv_dim,4,2,1)      # height/8, width/8
        self.conv4 = conv(4*conv_dim,8*conv_dim,4,2,1)      # height/16, width/16
        self.fc11 = nn.Linear(int(height/16)*int(width/16)*8*conv_dim,latent_size)
        self.fc12 = nn.Linear(int(height/16)*int(width/16)*8*conv_dim,latent_size)    

        #Decode
        self.fc3 = nn.Linear(latent_size,int(self.height/16)*int(self.width/16)*8*conv_dim)
        self.Tconv1 = Tconv(8*conv_dim,4*conv_dim,2,2,0)                            # height/8, widht/8
        self.Tconv2 = Tconv(4*conv_dim,4*conv_dim,2,2,0)                              # height/2, width/2
        self.Tconv3 = Tconv(4*conv_dim,2*conv_dim,2,2,0)
        self.Tconv4 = Tconv(2*conv_dim,nb_channels,2,2,0,batch_norm=False)            # height, width
        self.tanh = nn.Tanh()

    def encode(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.shape[0],-1)
        x1 = self.fc11(x)
        x2 = self.fc12(x)

        return x1, x2

    def decode(self, x):
        x = F.relu(self.fc3(x))
        x = x.view(x.shape[0], 8*self.conv_dim, int(self.height/16), int(self.width/16))
        x = F.relu(self.Tconv1(x))
        x = F.relu(self.Tconv2(x))
        x = F.relu(self.Tconv3(x))
        x = self.tanh(self.Tconv4(x))
        return x

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar


# def Pretrain_Classifier(nb_classes, transfer = None,hidden_units = [1024]):

#     if transfer == None or transfer == 'densenet121':
#         model = pretrain.densenet121(pretrained=True)
#         in_dim = 1024

#     if transfer == 'resnet101':
#         model = pretrain.densenet121(pretrained=True)
#         in_dim = 2048

#     if transfer == 'vgg19':
#         model = pretrain.vgg19(pretrained=True)
#         in_dim = 25088

#     for param in model.parameters():
#         param.requires_grad = False

#     Classifier = MLP(in_dim = in_dim, hidden_units=hidden_units, out_dim=nb_classes)
#     model.classifier = Classifier

#     return model

def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, layer_norm=True):
    """Creates a convolutional layer, with optional batch normalization.
    """
    layers = []
    conv_layer = nn.Conv2d(in_channels, out_channels, 
                           kernel_size, stride, padding, bias=True)
    
    # append conv layer
    layers.append(conv_layer)

    if layer_norm:
        # append batchnorm layer
        layers.append(nn.InstanceNorm2d(out_channels, affine=True))
     
    # using Sequential container
    return nn.Sequential(*layers)

def Tconv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    """Creates a transposed-convolutional layer, with optional batch normalization.
    """
    layers = []
    Tconv_layer = nn.ConvTranspose2d(in_channels, out_channels, 
                           kernel_size, stride, padding, bias=True)
    
    # append conv layer
    layers.append(Tconv_layer)

    if batch_norm:
        # append batchnorm layer
        layers.append(nn.BatchNorm2d(out_channels))
     
    # using Sequential container
    return nn.Sequential(*layers)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)