import numpy as np
from torchvision import transforms
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, DistributedSampler


IMAGE_SIZE = 224
ROTATIONAL_ANGLE = (90, 270)

class ImageAttributeDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        
        self.imageA_paths = self.dataframe['Image A']
        self.imageB_paths = self.dataframe['Image B']

        self.labels = self.dataframe['Similarity']


        self.image_transform = transforms.Compose([
              transforms.RandomRotation(degrees=ROTATIONAL_ANGLE),
              transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
              transforms.ToTensor(),
              transforms.Normalize( # Use when using pre-trained weights from TorchVision
                mean=[0.485, 0.456, 0.406],  # Mean values for normalization
                std=[0.229, 0.224, 0.225]     # Standard deviation values for normalization
                ),
          ])


    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        imageA_path = self.imageA_paths[idx]
        imageB_path = self.imageB_paths[idx]
        imageA = Image.open(imageA_path).convert('RGB')
        imageB = Image.open(imageB_path).convert('RGB')

        imageA = self.image_transform(imageA)
        imageB = self.image_transform(imageB)

        image_pair_label = torch.tensor(self.to_categorical(self.labels[idx])),

        return imageA, imageB, image_pair_label

    def to_categorical(self, label):
        mapped_label = label - 1
        num_classes = 4
        return np.eye(num_classes)[mapped_label]
        # return mapped_label

def main_dataset(df_train, df_val, batch_size, num_workers = 16):
    train_dataset = ImageAttributeDataset(df_train)
    val_dataset = ImageAttributeDataset(df_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True, num_workers=num_workers)
    val_loader =  DataLoader(val_dataset,batch_size=batch_size, num_workers=num_workers)
    return train_loader, val_loader