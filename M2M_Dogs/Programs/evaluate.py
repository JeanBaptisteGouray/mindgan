import torch
import torch.nn as nn

import numpy as np
from scipy import linalg


def calculate_activation_statistics(images, classifier):
    """
        Compute the parameters of the Gaussian which estimate the distribution
        of the genrated images
    """
    with torch.no_grad():
        if torch.cuda.is_available():
            images = images.cuda()
            classifier.cuda()
        act = classifier.forward(images)
    act = act.to('cpu').numpy()
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
        Compute the Frechet distance between two Gaussian distributions
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

def fid(images,fake_images,classifier):
    """
        Compute the FID
    """
    mu1, sigma1  = calculate_activation_statistics(images,classifier)
    mu2, sigma2  = calculate_activation_statistics(fake_images,classifier)
    fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6)
    return fid