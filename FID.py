import os
import warnings
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import json

import img_data as img_data
from img_data import CocoDataset
import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
from inception import InceptionV3
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from pycocotools.coco import COCO
from PIL import Image

def readfile(textpath):
    with open(textpath, 'r') as f:
        myList = json.load(f)
    return myList


def writefile(listdata, filepath):
    with open(filepath, 'w') as f:
        json.dump(listdata, f)
    return 0


def get_activations(images, model, batch_size=64, dims=2048, cuda=False, verbose=True):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- images      : Numpy array of dimension (n_images, 3, hi, wi). The values
                     must lie between 0 and 1.
    -- model       : Instance of inception model
    -- batch_size  : the images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size depends
                     on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    # d0 = images.shape[0]

    d0 = images.__len__() * batch_size
    if batch_size > d0:
        print(("Warning: batch size is bigger than the data size. " "Setting batch size to data size"))
        batch_size = d0

    n_batches = d0 // batch_size
    n_used_imgs = n_batches * batch_size

    pred_arr = np.empty((n_used_imgs, dims))
    for i, batch in enumerate(images):
        start = i * batch_size
        end = start + batch_size

        if cuda:
            batch = batch.cuda()

        pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.shape[2] != 1 or pred.shape[3] != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred_arr[start:end] = pred.cpu().data.numpy().reshape(batch_size, -1)

    if verbose:
        print(" done")

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representive data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representive data set.

    Returns:
    --   : The Frechet Distance.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ("fid calculation produces singular product; " "adding %s to diagonal of cov estimates") % eps
        # print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_activation_statistics(images, model, batch_size=64, dims=2048, cuda=False, verbose=True):
    """Calculation of the statistics used by the FID.
    Params:
    -- images      : Numpy array of dimension (n_images, 3, hi, wi). The values
                     must lie between 0 and 1.
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the
                     number of calculated batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(images, model, batch_size, dims, cuda, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def _compute_statistics_of_path(path, model, batch_size, dims, cuda):
    if path.endswith(".npz"):
        f = np.load(path, allow_pickle=True)
        m, s = f["mu"][:], f["sigma"][:]
        f.close()

    else:
        dataset = img_data.Dataset(
            path,
            transforms.Compose(
                [
                    transforms.Resize((299, 299)),
                    transforms.ToTensor(),
                ]
            ),
        )
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=8
        )
        m, s = calculate_activation_statistics(dataloader, model, batch_size, dims, cuda)

    return m, s


def get_groud_truth_images(file_name):
    # print(file_name)
    data = readfile(file_name)
    ground_truth_images = []
    for item in data:
        image_id = item['image_id']
        ground_truth_images.append(image_id)
    return ground_truth_images


def _compute_statistics_of_image(file_name, model, batch_size, dims, cuda):
    image_ids = get_groud_truth_images(file_name)
    # print(image_ids)
    dataDir = '/data1/data/coco/'
    dataType = 'val2017'
    dataset = CocoDataset(image_ids, dataDir, dataType, transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
        ]
    ))
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=8)
    m, s = calculate_activation_statistics(dataloader, model, batch_size, dims, cuda)

    return m, s


def calculate_fid_given_paths(imgpath, jsonpath, batch_size, cuda, dims):
    """Calculates the FID of two list of images"""

    if not os.path.exists(imgpath):
        raise RuntimeError("Invalid path: %s" % imgpath)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx])
    if cuda:
        model.cuda()

    m1, s1 = _compute_statistics_of_path(imgpath, model, batch_size, dims,
                                         cuda)  # function to process the generated images
    m2, s2 = _compute_statistics_of_image(jsonpath, model, batch_size, dims,
                                          cuda)  # function to process the ground truth images
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    return fid_value


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size to use")
    parser.add_argument(
        "--dims",
        type=int,
        default=2048,
        choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
        help=("Dimensionality of Inception features to use. " "By default, uses pool3 features"),
    )
    parser.add_argument("-c", "--gpu", default="6", type=str, help="GPU to use (leave blank for CPU only)")
    parser.add_argument("--path", type=str, required=True)
    # parser.add_argument("--jsonpath", type=str, required=True)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    path = '/ACTesting/test_data'

    files = ['new-captions_val2017_new', 'EC_captions_val2017', 'letter_captions_val2017', 'ER-R_captions_val2017', 'ER-A_captions_val2017','EC+ER-R_captions_val2017','EC+ER-A_captions_val2017']
    results = {}
    for item in files:
        
        json_name = item +'.json'
        args.jsonpath = os.path.join(path, json_name)
        image_path = os.path.join(args.path, item)
        fid_value = calculate_fid_given_paths(image_path, args.jsonpath, args.batch_size, args.gpu, args.dims).item()
        print(f"{item} FID: {fid_value}")
        results[item] = fid_value

    model_name = args.path.split('/')[-1]
    if model_name == '':
        model_name = args.path.split('/')[-2]
    file_name = os.path.join('/ACTesting/results/FID_'+ model_name + '.txt')
    with open(file_name, "w") as f:
        for k,v in results.items():
            f.write("{}: FID: {:.2f}\n".format(k, v))
