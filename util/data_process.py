import sys
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, STL10
from datasets_prep.lsun import LSUN
from datasets_prep.stackmnist_data import StackedMNIST, _data_transforms_stacked_mnist
from datasets_prep.lmdb_datasets import LMDBDataset
import random
import torch.utils.data as data
from torch.utils.data import ConcatDataset, Subset
    

class ConvertToRGB(object):
    def __call__(self, img):
        img_rgb = torch.cat([img, img, img], dim=0)
        return img_rgb

cmp = lambda x: transforms.Compose([*x])

def getCleanData(dataset, image_size=32):
    if dataset == 'cifar10':
        dataset = CIFAR10('./data', train=True, transform=transforms.Compose([
                        transforms.Resize(image_size),  # Resize images to your desired size
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]), download=True)
    
    elif dataset == 'cifar10_flip':
        dataset = CIFAR10('./data', train=True, transform=transforms.Compose([
                    transforms.Resize(image_size),  # Resize images to your desired size
                    transforms.RandomVerticalFlip(p=1.0),  # Flip all images vertically
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]), download=True)
        
    if dataset == 'mnist_cifar10':
        datasetcifar = CIFAR10('./data', train=True, transform=transforms.Compose([
                        transforms.Resize(image_size),  # Resize images to your desired size
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]), download=True)
        datasetcifar = [(img, -1) for img, label in datasetcifar if label == 0][:500]
        
        train_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.Grayscale(num_output_channels=3),  # Convert to "RGB" format (with duplicated channels)
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        datasetmnist = torchvision.datasets.MNIST(root='./data', train=True, transform=train_transform, download=True)

        class_1_images = [(img, label) for img, label in datasetmnist if label == 1][:5000]
        class_2_images = [(img, label) for img, label in datasetmnist if label == 2][:4000]
        class_0_images = [(img, label) for img, label in datasetmnist if label == 0][:500]

        dataset = ConcatDataset([
            datasetcifar,
            class_1_images,
            class_2_images,
            class_0_images
        ])

    if dataset == 'mnist12':
        train_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        datasetmnist = torchvision.datasets.MNIST(root='./data', train=True, transform=train_transform, download=True)
        class_1_images = [(img, label) for img, label in datasetmnist if label == 1][:5000]
        class_2_images = [(img, label) for img, label in datasetmnist if label == 2][:4500]
        class_0_images = [(img, label) for img, label in datasetmnist if label == 0][:500]

        dataset = ConcatDataset([
            class_1_images,
            class_2_images,
            class_0_images
        ])

    if dataset == 'mnist021':
        train_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        datasetmnist = torchvision.datasets.MNIST(root='./data', train=True, transform=train_transform, download=True)
        class_1_images = [(img, label) for img, label in datasetmnist if label == 1][:500]
        class_2_images = [(img, label) for img, label in datasetmnist if label == 2][:4500]
        class_0_images = [(img, label) for img, label in datasetmnist if label == 0][:5000]

        dataset = ConcatDataset([
            class_1_images,
            class_2_images,
            class_0_images
        ])

    elif dataset == 'mnist':
        
        train_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.Grayscale(num_output_channels=3),  # Convert to "RGB" format (with duplicated channels)
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=train_transform, download=True)

    elif dataset == 'fashion_mnist':
        
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),  # Convert to "RGB" format (with duplicated channels)
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=train_transform, download=True)

    elif dataset == 'mnist_1c':
        
        train_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=train_transform, download=True)

    elif dataset == 'fashion_mnist_1c':
        
        train_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        dataset = torchvision.datasets.FashionMNIST(root='./data/fashion_mnist', train=True, transform=train_transform, download=True)

    elif dataset == 'stl10':
        dataset = STL10('./data', split="unlabeled", transform=transforms.Compose([
                        transforms.Resize(64),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]), download=True)

    elif dataset == 'clipart':
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to your desired size
            transforms.CenterCrop((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Use torchvision's ImageFolder dataset to load your custom dataset
        dataset = torchvision.datasets.ImageFolder(root='./data/clipart', transform=train_transform)

    elif dataset == 'quickdraw':
        
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to your desired size
            transforms.CenterCrop((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Use torchvision's ImageFolder dataset to load your custom dataset
        dataset = torchvision.datasets.ImageFolder(root='./data/quickdraw', transform=train_transform)

    elif dataset == 'sketch':
        
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to your desired size
            transforms.CenterCrop((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Use torchvision's ImageFolder dataset to load your custom dataset
        dataset = torchvision.datasets.ImageFolder(root='./data/sketch', transform=train_transform)

    elif dataset == 'sketch_64':
        
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Use torchvision's ImageFolder dataset to load your custom dataset
        dataset = torchvision.datasets.ImageFolder(root='./data/sketch_64', transform=train_transform)

    elif dataset == 'clipart_64':
        
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Use torchvision's ImageFolder dataset to load your custom dataset
        dataset = torchvision.datasets.ImageFolder(root='./data/clipart_64', transform=train_transform)

    elif dataset == 'isolation_forest_celeba_5p_fashion':
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Use torchvision's ImageFolder dataset to load your custom dataset
        dataset = torchvision.datasets.ImageFolder(root='./data/isolation_forest_celeba_5p_fashion', transform=train_transform)

    elif dataset == 'stackmnist':
        train_transform, valid_transform = _data_transforms_stacked_mnist()
        dataset = StackedMNIST(root='./data', train=True, download=False, transform=train_transform)
        
    elif dataset == 'lsun':
        
        train_transform = transforms.Compose([
                        transforms.Resize((image_size, image_size)),
                        transforms.CenterCrop((image_size, image_size)),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                    ])

        train_data = LSUN(root='./data/LSUN/', classes=['church_outdoor_train'], transform=train_transform)
        subset = list(range(0, 120000))
        dataset = Subset(train_data, subset)
        

    elif dataset == 'celeba_256':
        # print(image_size)
        # print(type(image_size))
        train_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.CenterCrop((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
            ])
        
        dataset = LMDBDataset(root='./data/celeba-lmdb/', name='celeba', train=True, transform=train_transform)

    elif dataset == 'celeba_hq':
        # print(image_size)
        # print(type(image_size))
        train_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.CenterCrop((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
            ])
        
        dataset = LMDBDataset(root='./data/celeba-lmdb/', name='celeba', train=True, transform=train_transform)

    elif dataset == 'celeba_256_flip':
        # print(image_size)
        # print(type(image_size))
        train_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.CenterCrop((image_size, image_size)),
                transforms.RandomVerticalFlip(p=1.0),  # Flip all images vertically,
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
            ])
        
        dataset = LMDBDataset(root='./data/celeba-lmdb/', name='celeba', train=True, transform=train_transform)
        
    return dataset

def getMixedData(source_dataset, perturb_dataset, percentage = 0, image_size=32, random_seed = 19, shuffle = False):
    random.seed(random_seed)
    name_source = source_dataset
    source_dataset = getCleanData(source_dataset, image_size=image_size)
    perturb_dataset = getCleanData(perturb_dataset, image_size=image_size)
    if name_source in ['sketch', 'sketch_64']:
        num_samples = 30000
    else:
        num_samples = len(source_dataset)
    
    print(f'number of samples: {num_samples}')
    num_perturbed_samples = int(int(num_samples) * percentage / 100)
    
    num_source_samples = num_samples - num_perturbed_samples

    source_indices = random.sample(range(len(source_dataset)), num_source_samples)  # Randomly select indices of source data
    perturbed_indices = random.sample(range(len(perturb_dataset)), num_perturbed_samples)  # Randomly select indices of perturbed data

    print(f'source has {num_source_samples} data')
    print(f'perturb has {num_perturbed_samples} data')

    dataset = ConcatDataset([
        Subset(source_dataset, source_indices),
        Subset(perturb_dataset, perturbed_indices)
    ])

    if shuffle:
        indices = list(range(len(dataset)))
        random.shuffle(indices)

        # Create a new dataset with shuffled indices
        dataset = Subset(dataset, indices)

    return dataset

# ------------------------
# For Toy
# ------------------------
# datasets
class ToydatasetGaussian(data.Dataset):
    def __init__(self, cfg):
        self.dataset = torch.randn(cfg.num_data, cfg.data_dim) + torch.tensor([0,10])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]


class Toydatasetp(data.Dataset):
    def __init__(self, cfg):
        std = 0.5
        self.dataset = torch.cat([std*torch.randn(cfg.num_data//2, cfg.data_dim)+1, 
                                  std*torch.randn(cfg.num_data-cfg.num_data//2, cfg.data_dim)-1])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]


class Toydatasetq(data.Dataset):
    def __init__(self, cfg):
        std = 0.5
        self.dataset = torch.cat([std*torch.randn(2*cfg.num_data//3, cfg.data_dim)+2, 
                                  std*torch.randn(cfg.num_data-2*cfg.num_data//3, cfg.data_dim)-1])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]


class ToydatasetOutlier(data.Dataset):
    def __init__(self, cfg):
        M = int(cfg.num_data*cfg.p)
        self.dataset = torch.cat([0.1*torch.randn(cfg.num_data-M, cfg.data_dim) + 1, 0.05*torch.randn(M, cfg.data_dim) - 1])
        total_samples = self.dataset.size(0)

        # Generate a random permutation of indices
        random_indices = torch.randperm(total_samples)

        # Use the random indices to shuffle the dataset
        self.dataset = self.dataset[random_indices]
        # self.dataset = torch.cat([0.1*torch.randn(cfg.num_data-M, cfg.data_dim) + 2, 0.05*torch.randn(M, cfg.data_dim) - 2])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]
    
class ToydatasetNoise(data.Dataset):
    def __init__(self, cfg):
        self.N = cfg.num_data
        self.dim = cfg.data_dim
    
    def __len__(self):
        return int(self.N)
        
    
    def __getitem__(self, idx):
        return torch.randn((1, self.dim))
        # torch.randn((1, self.dim))

def get_datasets(cfg):
    src_name, tar_name = cfg.source_name, cfg.target_name
    datasets = []

    for name in [src_name, tar_name]:
        if name == 'gaussian':
            dataset = ToydatasetGaussian(cfg)
        elif name == 'p':
            dataset = Toydatasetp(cfg)
        elif name == 'q':
            dataset = Toydatasetq(cfg)
        elif name == 'outlier':
            dataset = ToydatasetOutlier(cfg)
        elif name == 'noise':
            dataset = ToydatasetNoise(cfg)
        else:
            raise NotImplementedError
        
        datasets.append(dataset)
    
    return datasets
