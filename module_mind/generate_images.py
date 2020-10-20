import torch
import torch.nn as nn 
from torchvision import transforms, models, datasets
from torchvision.utils import save_image
import numpy as np
from scipy import linalg
import json
import os
from tqdm import tqdm

from module_mind.models import FFHQGenerator

from defaults import get_cfg_defaults
import logging
from checkpointer import Checkpointer
from model import Model


nombre_images=4
batch_size=4
out_path='module_mind/images_generated/mind2mind/'

hyperparameters_path = 'module_mind/output/submitted/parameters.json'
model_m2m_path = 'module_mind/output/submitted/model299.pth'

num_workers = 8
pin_memory = True
    
print('load hyperparameters')
with open(hyperparameters_path, 'r') as f:
    hyperparameters = json.load(f)

    
print('load M2M')
m2m_generator = FFHQGenerator(hyperparameters['z_dim'], hyperparameters['hidden_generator_c'])

save_dict = torch.load(model_m2m_path)
state_dict = save_dict['Generator']

m2m_generator.load_state_dict(state_dict)
m2m_generator.eval()

if torch.cuda.is_available():
    m2m_generator.cuda()

torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.cuda.current_device()

cfg = get_cfg_defaults()

config_file = 'configs/ffhq.yaml'
if len(os.path.splitext(config_file)[1]) == 0:
    config_file += '.yaml'

if not os.path.exists(config_file) and os.path.exists(os.path.join('configs', config_file)):
    config_file = os.path.join('configs', config_file)
cfg.merge_from_file(config_file)
cfg.freeze()

torch.cuda.set_device(0)

model = Model(
    startf=cfg.MODEL.START_CHANNEL_COUNT,
    layer_count=cfg.MODEL.LAYER_COUNT,
    maxf=cfg.MODEL.MAX_CHANNEL_COUNT,
    latent_size=cfg.MODEL.LATENT_SPACE_SIZE,
    truncation_psi=cfg.MODEL.TRUNCATIOM_PSI,
    truncation_cutoff=cfg.MODEL.TRUNCATIOM_CUTOFF,
    mapping_layers=cfg.MODEL.MAPPING_LAYERS,
    channels=cfg.MODEL.CHANNELS,
    generator=cfg.MODEL.GENERATOR,
    encoder=cfg.MODEL.ENCODER)
model.cuda(0)
model.eval()
model.requires_grad_(False)

decoder = model.decoder
encoder = model.encoder
mapping_tl = model.mapping_tl
mapping_fl = model.mapping_fl
dlatent_avg = model.dlatent_avg


logger = logging.getLogger("logger")
logger.setLevel(logging.DEBUG)

model_dict = {
    'discriminator_s': encoder,
    'generator_s': decoder,
    'mapping_tl_s': mapping_tl,
    'mapping_fl_s': mapping_fl,
    'dlatent_avg': dlatent_avg
}

checkpointer = Checkpointer(cfg,
                            model_dict,
                            {},
                            logger=logger,
                            save=False)

extra_checkpoint_data = checkpointer.load()

model.eval()
layer_count = cfg.MODEL.LAYER_COUNT


def decode(x):
    decoded = []
    for i in range(x.shape[0]):
        r = model.decoder(x[i][None, ...], layer_count - 1, 1, noise=True)
        decoded.append(r)
    return torch.cat(decoded)


def generate_images(path,name, batch_size, hyperparameters, generator):
    z = torch.from_numpy(np.random.uniform(-1, 1,size=(batch_size,hyperparameters['z_dim']))).float()
    if torch.cuda.is_available():
        z = z.cuda()

    fake_images = decode(generator.sample(z))
    for i in range(batch_size):
        save_image(fake_images[i],path+name+str(i)+'.png', normalize=True, range=(-1,1))
    
if not os.path.exists(out_path):
    os.makedirs(out_path)

nombre_batches=nombre_images//batch_size

for j in tqdm(range(nombre_batches)):
    name=str(j)+'_'
    generate_images(out_path, name, batch_size,  hyperparameters, m2m_generator)



