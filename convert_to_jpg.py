import sys
import torchvision
import torchvision.transforms as transforms
from util.data_process import getCleanData
from torch.utils.data import ConcatDataset
import os


# dataset = STL10('./data', split="unlabeled", transform=transforms.Compose([
#                 transforms.Resize(64),
#                 transforms.ToTensor()]), download=True)

# train_transform = transforms.Compose([
#                 transforms.Resize((64, 64)),
#                 transforms.CenterCrop((64, 64)),
#                 transforms.ToTensor()
#             ])
        
# dataset = LMDBDataset(root='./data/celeba-lmdb/', name='celeba', train=True, transform=train_transform)

# train_transform = transforms.Compose([
#     transforms.Resize(32),
#     transforms.ToTensor(),
# ])

# dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=train_transform, download=True)

train_transform = transforms.Compose([
    transforms.Resize(32),
    transforms.Grayscale(num_output_channels=3),  # Convert to "RGB" format (with duplicated channels)
    transforms.ToTensor(),
])
datasetmnist = torchvision.datasets.MNIST(root='./data', train=True, transform=train_transform, download=True)
class_1_images = [(img, label) for img, label in datasetmnist if label == 1][:5000]
class_2_images = [(img, label) for img, label in datasetmnist if label == 2][:4000]
class_0_images = [(img, label) for img, label in datasetmnist if label == 0][:500]

dataset = ConcatDataset([
    class_1_images,
    class_2_images,
    class_0_images
])

def organizeData(dataset, saved_folder):
    # Create a directory for STL10 if it doesn't exist
    data_dir = f'./{saved_folder}'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Loop through the dataset and move files to their respective class folders
    for i, (_, class_id) in enumerate(dataset):
        class_folder = os.path.join(data_dir, str(class_id))
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)
        file_name = f'image_{i}.jpg'  # You may want to adjust the file name as needed
        file_path = os.path.join(class_folder, file_name)
        torchvision.utils.save_image(dataset[i][0], file_path)
        # shutil.copy(dataset.data[i], file_path)  # Copy the file to the class folder

# Call the function to organize the dataset
organizeData(dataset, './images_for_fid/mnist_3c_120/')
