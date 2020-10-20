import dlutils
import zipfile
import os

import logging
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import pickle as pkl 

from model import Model
from defaults import get_cfg_defaults
from checkpointer import Checkpointer

# https://drive.google.com/drive/folders/11Vz0fqHS2rXDb5pprgTjpD7S2BAJhi1P
if not os.path.exists('module_mind/Dataset/Celeba-HQ/data1024x1024.zip'):
    dlutils.download.from_google_drive('1-LFFkFKNuyBO1sjkM4t_AArIXr3JAOyl', directory='module_mind/Dataset/Celeba-HQ')

if not os.path.exists('module_mind/Dataset/Celeba-HQ/data1024x1024'):
    with zipfile.ZipFile('module_mind/Dataset/Celeba-HQ/data1024x1024.zip', 'r') as zip_ref:
        zip_ref.extractall('module_mind/Dataset/Celeba-HQ/')

if not os.path.exists('training_artifacts/ffhq/model_submitted.pth'):
    
    try:
        dlutils.download.from_google_drive('170Qldnn28IwnVm9CQEq1AZhVsK7PJ0Xz', directory='training_artifacts/ffhq')
    except IOError:
        dlutils.download.from_url('https://alaeweights.s3.us-east-2.amazonaws.com/ffhq/model_submitted.pth', directory='../training_artifacts/ffhq')

with open('training_artifacts/ffhq/last_checkpoint','w') as f:
    f.write('training_artifacts/ffhq/model_submitted.pth')


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
im_size = 2 ** (cfg.MODEL.LAYER_COUNT + 1)
layer_count = cfg.MODEL.LAYER_COUNT

images = os.listdir('module_mind/Dataset/Celeba-HQ/data1024x1024')[:29000]

for image in tqdm(images):
    im = np.asarray(Image.open('module_mind/Dataset/Celeba-HQ/data1024x1024/' + image))
    im = im.transpose((2, 0, 1))

    x = torch.tensor(np.asarray(im, dtype=np.float32), requires_grad=True).cuda() / 127.5 - 1.
    if x.shape[0] == 4:
        x = x[:3]

    factor = x.shape[2] // im_size
    if factor != 1:
        print(factor)
        x = torch.nn.functional.avg_pool2d(x[None, ...], factor, factor)[0]
    
    assert x.shape[2] == im_size

    x = x.unsqueeze(0)

    Z, _ = model.encode(x[0][None, ...], layer_count - 1, 1)

    latent = Z.view(512).cpu().detach().numpy()

    assert latent.shape[0] == 512
    
    if not os.path.exists('module_mind/Dataset/Encoded_from_FFHQ'):
        os.makedirs('module_mind/Dataset/Encoded_from_FFHQ')

    with open('module_mind/Dataset/Encoded_from_FFHQ/'+image.replace('.jpg',''), 'wb') as f:
        pkl.dump(latent,f)
