import subprocess
import os
import abc
import hashlib
import zipfile
import glob
import logging
import tarfile
from skimage.io import imread
from PIL import Image
from tqdm import tqdm
import numpy as np
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import h5py
from sys import getsizeof

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets


DIR = os.path.abspath(os.path.dirname(__file__))
COLOUR_BLACK = 0
COLOUR_WHITE = 1
DATASETS_DICT = {"mnist": "MNIST",
                 "fashion": "FashionMNIST",
                 "dsprites": "DSprites",
                 "celeba": "CelebA",
                 "chairs": "Chairs",
                 "cifar10": "CIFAR10",
                 "cifar100": "CIFAR100",
                 "3dshapes":"Shapes3D",
                 "mpi3dtoy":"MPI3DToy"}

DATASETS = list(DATASETS_DICT.keys())

def get_dataset(dataset):
    """Return the correct dataset."""
    dataset = dataset.lower()
    try:
        # eval because stores name as string in order to put it at top of file
        return eval(DATASETS_DICT[dataset])
    except KeyError:
        raise ValueError("Unkown dataset: {}".format(dataset))


def get_img_size(dataset):
    """Return the correct image size."""
    return get_dataset(dataset).img_size


def get_background(dataset):
    """Return the image background color."""
    return get_dataset(dataset).background_color


def get_dataloaders(dataset, root=None, shuffle=True, pin_memory=True,
                    batch_size=128, val_split=0.1, logger=logging.getLogger(__name__), **kwargs):
    """A generic data loader with added train/val split support

    Parameters
    ----------
    dataset : {"mnist", "fashion", "dsprites", "celeba", "chairs"}
        Name of the dataset to load
    val_split : float
        Fraction of the dataset to use for validation (default 0.1)
    root : str
        Path to the dataset root. If `None` uses the default one.
    kwargs :
        Additional arguments to `DataLoader`. Default values are modified.
    """
    pin_memory = pin_memory and torch.cuda.is_available  # only pin if GPU available
    Dataset = get_dataset(dataset)
    dataset_obj = Dataset(logger=logger) if root is None else Dataset(root=root, logger=logger)

    # Split indices for train/val
    n_total = len(dataset_obj)
    n_val = int(val_split * n_total)
    n_train = n_total - n_val
    train_set, val_set = torch.utils.data.random_split(dataset_obj, [n_train, n_val])

    train_loader = DataLoader(train_set,
                      batch_size=batch_size,
                      shuffle=True,
                      pin_memory=pin_memory,
                      **kwargs)
    val_loader = DataLoader(val_set,
                      batch_size=batch_size,
                      shuffle=False,
                      pin_memory=pin_memory,
                      **kwargs)
    # For compatibility, also return the full dataset and a viz/test loader
    viz_loader = DataLoader(dataset_obj, batch_size=1, shuffle=True)
    test_loader = DataLoader(dataset_obj, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, dataset_obj, viz_loader, test_loader




class DisentangledDataset(Dataset, abc.ABC):
    """Base Class for disentangled VAE datasets.

    Parameters
    ----------
    root : string
        Root directory of dataset.

    transforms_list : list
        List of `torch.vision.transforms` to apply to the data when loading it.
    """

    def __init__(self, root, transforms_list=[], logger=logging.getLogger(__name__)):
        self.root = root
        self.train_data = os.path.join(root, type(self).files["train"])
        self.transforms = transforms.Compose(transforms_list)
        self.logger = logger

        if not os.path.isdir(root):
            self.logger.info("Downloading {} ...".format(str(type(self))))
            self.download()
            self.logger.info("Finished Downloading.")

    def __len__(self):
        return len(self.imgs)

    @abc.abstractmethod
    def __getitem__(self, idx):
        """Get the image of `idx`.

        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `img_size`.
        """
        pass

    @abc.abstractmethod
    def download(self):
        """Download the dataset. """
        pass

class Shapes3D(DisentangledDataset):
    """Shapes3D Dataset from [1].

    Disentanglement test Toy dataset. Procedurally generated 3D shapes, from 7
    disentangled latent factors. This dataset uses 7 latents, controlling the object color,
    object shape, object size, camera height, background color, horizontal axis and vertical axis of a sprite. All possible variations of
    the latents are present. Ordering along dimension 1 is fixed and can be mapped
    back to the exact latent values that generated that image. Pixel outputs are
    different. No noise added.

    Notes
    -----
    - Link : https://github.com/deepmind/3d-shapes
    - hard coded metadata because issue with python 3 loading of python 2

    Parameters
    ----------
    root : string
        Root directory of dataset.

    References
    ----------
    [1]http://proceedings.mlr.press/v80/kim18b.html

    """
    urls = {"train": "https://storage.cloud.google.com/3d-shapes/3dshapes.h5"}
    files = {"train": "3dshapes.h5"}
    lat_names = ('floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape',
                     'orientation')
    lat_sizes = np.array([10, 10, 10, 8, 4, 15])
    img_size = (3, 64, 64)
    background_color = COLOUR_WHITE
    

    def __init__(self, root=os.path.join(DIR, '../data/3dshapes/'), **kwargs):
        super().__init__(root, [transforms.ToTensor()], **kwargs)  
        with h5py.File(self.train_data, 'r') as dataset_zip:
            self.imgs = dataset_zip['images'][()]
            self.lat_values = dataset_zip['labels'][()]
        
        print("Images memory usage: ", round(getsizeof(self.imgs) / 1024 / 1024,2))
    def download(self):
        """Download the dataset."""
        os.makedirs(self.root)
        subprocess.check_call(["curl", "-L", type(self).urls["train"],
                               "--output", self.train_data])

    def __getitem__(self, idx):
        """Get the image of `idx`
        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `img_size`.

        lat_value : np.array
            Array of length 6, that gives the value of each factor of variation.
        """
        # stored image have binary and shape (H x W) so multiply by 255 to get pixel
        # values + add dimension
        im = self.imgs[idx]
        # ToTensor transforms numpy.ndarray (H x W x C) in the range
        # [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        im = self.transforms(im)

        lat_value = self.lat_values[idx]
        return im, lat_value

    def images_from_data_gen(self, sample_size, y, y_lat):
        #sample two sets of data generative factors such that the yth value is the same accross the two sets

        samples = np.zeros((sample_size, self.lat_sizes.size))

        for i, lat_size in enumerate(self.lat_sizes):
            samples[:, i] = y_lat if i == y else np.random.randint(lat_size, size=sample_size) 
        latents_bases = np.concatenate((self.lat_sizes[::-1].cumprod()[::-1][1:],
                                np.array([1,])))

        latent_indices = np.dot(samples, latents_bases).astype(int)

        ims = []
        for ind in latent_indices:
            im = self.imgs[ind]
            im = np.asarray(im)
            im = self.transforms(im)
            ims.append(im)
        
        ims = np.stack(ims, axis=0)
        return torch.from_numpy(ims) 
        


     
        #not enough RAM :(
class MPI3DToy(DisentangledDataset):
    """MPI3D Dataset from [1].

    Disentanglement test Toy dataset. Procedurally generated 3D shapes, from 7
    disentangled latent factors. This dataset uses 7 latents, controlling the object color,
    object shape, object size, camera height, background color, horizontal axis and vertical axis of a sprite. All possible variations of
    the latents are present. Ordering along dimension 1 is fixed and can be mapped
    back to the exact latent values that generated that image. Pixel outputs are
    different. No noise added.

    Notes
    -----
    - Link : https://github.com/rr-learning/disentanglement_dataset
    - hard coded metadata because issue with python 3 loading of python 2

    Parameters
    ----------
    root : string
        Root directory of dataset.

    References
    ----------
    [1] Higgins, I., Matthey, L., Pal, A., Burgess, C., Glorot, X., Botvinick,
        M., ... & Lerchner, A. (2017). beta-vae: Learning basic visual concepts
        with a constrained variational framework. In International Conference
        on Learning Representations.

    """
    urls = {"train": "https://storage.googleapis.com/disentanglement_dataset/Final_Dataset/mpi3d_toy.npz"}
    files = {"train": "mpi3d_toy.npz"}
    lat_names = ('object_color', 'object_shape', 'object_size', 'camera_height', 'background_color', 'horizontal_axis', 'vertical_axis')
    lat_sizes = np.array([6, 6, 2, 3, 3, 40, 40])
    img_size = (3, 64, 64)
    background_color = COLOUR_BLACK
    

    def __init__(self, root=os.path.join(DIR, '../data/mpi3dtoy/'), **kwargs):
        super().__init__(root, [transforms.ToTensor()], **kwargs)
        dataset_zip = np.load(self.train_data)
        self.imgs = dataset_zip['images']
        print("Images memory usage:", round(getsizeof(self.imgs) / 1024 / 1024,2))
      
        
    def download(self):
        """Download the dataset."""
        os.makedirs(self.root)
        subprocess.check_call(["curl", "-L", type(self).urls["train"],
                               "--output", self.train_data])

    def __getitem__(self, idx):
        """Get the image of `idx`
        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `img_size`.

        lat_value : np.array
            Array of length 6, that gives the value of each factor of variation.
        """
        # ToTensor transforms numpy.ndarray (H x W x C) in the range
        # [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        im = self.imgs[idx]
        # ToTensor transforms numpy.ndarray (H x W x C) in the range
        # [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        im = self.transforms(im)

        #lat_value = self.lat_values[idx]
        return im, 0

    def images_from_data_gen(self, sample_size, y, y_lat):
        #sample two sets of data generative factors such that the yth value is the same accross the two sets

        samples = np.zeros((sample_size, self.lat_sizes.size))

        for i, lat_size in enumerate(self.lat_sizes):
            samples[:, i] = y_lat if i == y else np.random.randint(lat_size, size=sample_size) 
        latents_bases = np.concatenate((self.lat_sizes[::-1].cumprod()[::-1][1:],
                                np.array([1,])))

        latent_indices = np.dot(samples, latents_bases).astype(int)

        ims = []
        for ind in latent_indices:
            im = self.imgs[ind]
            im = np.asarray(im)
            im = self.transforms(im)
            ims.append(im)
        
        ims = np.stack(ims, axis=0)
        return torch.from_numpy(ims)

class DSprites(DisentangledDataset):
    """DSprites Dataset from [1].

    Disentanglement test Sprites dataset.Procedurally generated 2D shapes, from 6
    disentangled latent factors. This dataset uses 6 latents, controlling the color,
    shape, scale, rotation and position of a sprite. All possible variations of
    the latents are present. Ordering along dimension 1 is fixed and can be mapped
    back to the exact latent values that generated that image. Pixel outputs are
    different. No noise added.

    Notes
    -----
    - Link : https://github.com/deepmind/dsprites-dataset/
    - hard coded metadata because issue with python 3 loading of python 2

    Parameters
    ----------
    root : string
        Root directory of dataset.

    References
    ----------
    [1] Higgins, I., Matthey, L., Pal, A., Burgess, C., Glorot, X., Botvinick,
        M., ... & Lerchner, A. (2017). beta-vae: Learning basic visual concepts
        with a constrained variational framework. In International Conference
        on Learning Representations.

    """
    urls = {"train": "https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz?raw=true"}
    files = {"train": "dsprite_train.npz"}
    lat_names = ('shape', 'scale', 'orientation', 'posX', 'posY')
    background_color = COLOUR_BLACK
    lat_sizes = np.array([3, 6, 40, 32, 32])
    img_size = (1, 64, 64)
    lat_values = {'posX': np.array([0., 0.03225806, 0.06451613, 0.09677419, 0.12903226,
                                    0.16129032, 0.19354839, 0.22580645, 0.25806452,
                                    0.29032258, 0.32258065, 0.35483871, 0.38709677,
                                    0.41935484, 0.4516129, 0.48387097, 0.51612903,
                                    0.5483871, 0.58064516, 0.61290323, 0.64516129,
                                    0.67741935, 0.70967742, 0.74193548, 0.77419355,
                                    0.80645161, 0.83870968, 0.87096774, 0.90322581,
                                    0.93548387, 0.96774194, 1.]),
                  'posY': np.array([0., 0.03225806, 0.06451613, 0.09677419, 0.12903226,
                                    0.16129032, 0.19354839, 0.22580645, 0.25806452,
                                    0.29032258, 0.32258065, 0.35483871, 0.38709677,
                                    0.41935484, 0.4516129, 0.48387097, 0.51612903,
                                    0.5483871, 0.58064516, 0.3658, 0.96664389, 1.12775121,
                                           1.28885852, 1.44996584, 1.61107316, 1.77218047,
                                           1.93328779, 2.0943951, 2.25550242, 2.41660973,
                                           2.57771705, 2.73882436, 2.89993168, 3.061039,
                                           3.22214631, 3.38325363, 3.54436094, 3.70546826,
                                           3.86657557, 4.02768289, 4.1887902, 4.34989752,
                                           4.51100484, 4.67211215, 4.83321947, 4.99432678,
                                           5.1554341, 5.31654141, 5.47764873, 5.63875604,
                                           5.79986336, 5.96097068, 6.12207799, 6.28318531]),
                  'shape': np.array([1., 2., 3.]),
                  'color': np.array([1.])}

    def __init__(self, root=os.path.join(DIR, '../data/dsprites/'), **kwargs):
        super().__init__(root, [transforms.ToTensor()], **kwargs)

        dataset_zip = np.load(self.train_data)
        self.imgs = dataset_zip['imgs']
        self.lat_values = dataset_zip['latents_values']

    def download(self):
        """Download the dataset."""
        os.makedirs(self.root)
        subprocess.check_call(["curl", "-L", type(self).urls["train"],
                               "--output", self.train_data])

    def __getitem__(self, idx):
        """Get the image of `idx`
        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `img_size`.

        lat_value : np.array
            Array of length 6, that gives the value of each factor of variation.
        """
        # stored image have binary and shape (H x W) so multiply by 255 to get pixel
        # values + add dimension
        sample = np.expand_dims(self.imgs[idx] * 255, axis=-1)

        # ToTensor transforms numpy.ndarray (H x W x C) in the range
        # [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        sample = self.transforms(sample)

        lat_value = self.lat_values[idx]
        return sample, lat_value

    def images_from_data_gen(self, sample_size, y, y_lat):
        """
        Compute the disentanglement metric score as proposed in the original paper
        reference: https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_reloading_example.ipynb
        """
        #sample two sets of data generative factors such that the yth value is the same accross the two sets
        samples = np.zeros((sample_size, self.lat_sizes.size))

        for i, lat_size in enumerate(self.lat_sizes):
            samples[:, i] = y_lat if i == y else np.random.randint(lat_size, size=sample_size) 
        latents_bases = np.concatenate((self.lat_sizes[::-1].cumprod()[::-1][1:],
                                np.array([1,])))

        latent_indices = np.dot(samples, latents_bases).astype(int)
    
        #use the data generative factors to simulate two sets of images from the dataset
        imgs_sampled = torch.from_numpy(self.imgs[latent_indices]).unsqueeze_(1).float()

        #print(imgs_sampled.shape)
        return imgs_sampled


class CelebA(DisentangledDataset):
    """CelebA Dataset from [1].

    CelebFaces Attributes Dataset (CelebA) is a large-scale face attributes dataset
    with more than 200K celebrity images, each with 40 attribute annotations.
    The images in this dataset cover large pose variations and background clutter.
    CelebA has large diversities, large quantities, and rich annotations, including
    10,177 number of identities, and 202,599 number of face images.

    Notes
    -----
    - Link : http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

    Parameters
    ----------
    root : string
        Root directory of dataset.

    References
    ----------
    [1] Liu, Z., Luo, P., Wang, X., & Tang, X. (2015). Deep learning face
        attributes in the wild. In Proceedings of the IEEE international conference
        on computer vision (pp. 3730-3738).

    """
    urls = {"train": "https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip"}
    files = {"train": "img_align_celeba"}
    img_size = (3, 64, 64)
    background_color = COLOUR_WHITE

    def __init__(self, root=os.path.join(DIR, '../data/celeba'), **kwargs):
        super().__init__(root, [transforms.ToTensor()], **kwargs)

        self.imgs = glob.glob(self.train_data + '/*')

    def download(self):
        """Download the dataset."""
        save_path = os.path.join(self.root, 'celeba.zip')
        os.makedirs(self.root)
        subprocess.check_call(["curl", "-L", type(self).urls["train"],
                               "--output", save_path])

        hash_code = '00d2c5bc6d35e252742224ab0c1e8fcb'
        assert hashlib.md5(open(save_path, 'rb').read()).hexdigest() == hash_code, \
            '{} file is corrupted.  Remove the file and try again.'.format(save_path)

        with zipfile.ZipFile(save_path) as zf:
            self.logger.info("Extracting CelebA ...")
            zf.extractall(self.root)

        os.remove(save_path)

        self.logger.info("Resizing CelebA ...")
        preprocess(self.train_data, size=type(self).img_size[1:])

    def __getitem__(self, idx):
        """Get the image of `idx`

        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `img_size`.

        placeholder :
            Placeholder value as their are no targets.
        """
        img_path = self.imgs[idx]
        # img values already between 0 and 255
        img = imread(img_path)

        # put each pixel in [0.,1.] and reshape to (C x H x W)
        img = self.transforms(img)

        # no label so return 0 (note that can't return None because)
        # dataloaders requires so
        return img, 0

class Chairs(datasets.ImageFolder):
    """Chairs Dataset from [1].

    Notes
    -----
    - Link : https://www.di.ens.fr/willow/research/seeing3Dchairs

    Parameters
    ----------
    root : string
        Root directory of dataset.

    References
    ----------
    [1] Aubry, M., Maturana, D., Efros, A. A., Russell, B. C., & Sivic, J. (2014).
        Seeing 3d chairs: exemplar part-based 2d-3d alignment using a large dataset
        of cad models. In Proceedings of the IEEE conference on computer vision
        and pattern recognition (pp. 3762-3769).

    """
    urls = {"train": "https://www.di.ens.fr/willow/research/seeing3Dchairs/data/rendered_chairs.tar"}
    files = {"train": "chairs_64"}
    img_size = (1, 64, 64)
    background_color = COLOUR_WHITE
    def __init__(self, root=os.path.join(DIR, '../data/chairs'),
                 logger=logging.getLogger(__name__)):
        self.root = root
        self.train_data = os.path.join(root, type(self).files["train"])
        self.transforms = transforms.Compose([transforms.Grayscale(),
                                              transforms.ToTensor()])
        self.logger = logger

        if not os.path.isdir(root):
            self.logger.info("Downloading {} ...".format(str(type(self))))
            self.download()
            self.logger.info("Finished Downloading.")

        super().__init__(self.train_data, transform=self.transforms)

    def download(self):
        """Download the dataset."""
        save_path = os.path.join(self.root, 'chairs.tar')
        os.makedirs(self.root)
        subprocess.check_call(["curl", type(self).urls["train"],
                               "--output", save_path])

        self.logger.info("Extracting Chairs ...")
        tar = tarfile.open(save_path)
        tar.extractall(self.root)
        tar.close()
        os.rename(os.path.join(self.root, 'rendered_chairs'), self.train_data)

        os.remove(save_path)

        self.logger.info("Preprocessing Chairs ...")
        preprocess(os.path.join(self.train_data, '*/*'),  # root/*/*/*.png structure
                   size=type(self).img_size[1:],
                   center_crop=(400, 400))


class MNIST(datasets.MNIST):
    """Mnist wrapper. Docs: `datasets.MNIST.`"""
    img_size = (1, 64, 64)
    background_color = COLOUR_BLACK
    def __init__(self, root=os.path.join(DIR, '../data/mnist'), **kwargs):
        super().__init__(root,
                         train=True,
                         download=True,
                         transform=transforms.Compose([
                             transforms.Resize(64),
                             transforms.ToTensor()
                         ]))

class CIFAR10(datasets.CIFAR10):
    """CIFAR10 wrapper"""
    img_size = (3, 64, 64)
    background_color = COLOUR_BLACK
    
    def __init__(self, root=os.path.join(DIR, '../data/cifar10'), **kwargs):
        super().__init__(root,
                         train=True,
                         download=True,
                         transform=transforms.Compose([
                             transforms.Resize(64),
                             transforms.ToTensor()
                         ]))

class CIFAR100(datasets.CIFAR100):
    """CIFAR100 wrapper"""
    img_size = (3, 64, 64)
    background_color = COLOUR_BLACK
    
    def __init__(self, root=os.path.join(DIR, '../data/cifar100'), **kwargs):
        super().__init__(root,
                         train=True,
                         download=True,
                         transform=transforms.Compose([
                             transforms.Resize(64),
                             transforms.ToTensor()
                         ]))

class FashionMNIST(datasets.FashionMNIST):
    """Fashion Mnist wrapper. Docs: `datasets.FashionMNIST.`"""
    img_size = (1, 64, 64)
    background_color = COLOUR_BLACK
    def __init__(self, root=os.path.join(DIR, '../data/fashionMnist'), **kwargs):
        super().__init__(root,
                         train=True,
                         download=True,
                         transform=transforms.Compose([
                             transforms.Resize(64),
                             transforms.ToTensor()
                         ]))


# HELPERS
def preprocess(root, size=(64, 64), img_format='JPEG', center_crop=None):
    """Preprocess a folder of images.

    Parameters
    ----------
    root : string
        Root directory of all images.

    size : tuple of int
        Size (width, height) to rescale the images. If `None` don't rescale.

    img_format : string
        Format to save the image in. Possible formats:
        https://pillow.readthedocs.io/en/3.1.x/handbook/image-file-formats.html.

    center_crop : tuple of int
        Size (width, height) to center-crop the images. If `None` don't center-crop.
    """
    imgs = []
    for ext in [".png", ".jpg", ".jpeg"]:
        imgs += glob.glob(os.path.join(root, '*' + ext))

    for img_path in tqdm(imgs):
        img = Image.open(img_path)
        width, height = img.size

        if size is not None and width != size[1] or height != size[0]:
            img = img.resize(size, Image.ANTIALIAS)

        if center_crop is not None:
            new_width, new_height = center_crop
            left = (width - new_width) // 2
            top = (height - new_height) // 2
            right = (width + new_width) // 2
            bottom = (height + new_height) // 2

            img.crop((left, top, right, bottom))

        img.save(img_path, img_format)
