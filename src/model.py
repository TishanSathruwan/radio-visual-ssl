import torch
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model
import torch.nn.init as init
from torchvision.models import resnet18

# ---------------------------------------------------- Davit model -----------------------------------------------

class davit(nn.Module):
    def __init__(self):
        super(davit, self).__init__()
        # Load the pre-trained davit model
        davit_model = create_model('davit_small.msft_in1k', pretrained=True, num_classes=2)

        # Remove the last fully connected layer (classifier)
        # checkpoint = torch.load(DAVIT_PATH, map_location=DEVICE)

        # davit_model.load_state_dict(checkpoint, strict=False)
        # davit_model.classifier = nn.Identity()

        self.features = nn.Sequential(*list(davit_model.children())[:-1])
        # self.pool1 = nn.MaxPool2d(kernel_size=1, stride=1)

    def forward(self, x):
        x = self.features(x)
        embedding = x
        # embedding = self.pool1(x)
        # x = F.adaptive_avg_pool2d(x, (1, 1))   # To get desired dimension
        # x = x.view(x.size(0), -1)
        return embedding
# ---------------------------------------------------- Xception model -----------------------------------------------

class Xception(nn.Module):
    def __init__(self):
        super(Xception, self).__init__()
        # Load the pre-trained tinyViT model
        xception_model = create_model('xception41', pretrained=True, num_classes=2)

        self.features = nn.Sequential(*list(xception_model.children())[:-1])
        self.pool1 = nn.AvgPool2d(kernel_size=1, stride=1)

    def forward(self, x):
        x = self.features(x)
        embedding = self.pool1(x)
        return embedding

# ---------------------------------------------------- TinyViT model -----------------------------------------------

class tinyViT(nn.Module):
    def __init__(self):
        super(tinyViT, self).__init__()
        # Load the pre-trained tinyViT model
        tinyvit_model = create_model('tiny_vit_21m_512', pretrained=True, num_classes=2)

        self.features = nn.Sequential(*list(tinyvit_model.children())[:-1])
        self.pool1 = nn.AvgPool2d(kernel_size=1, stride=1)

    def forward(self, x):
        x = self.features(x)
        embedding = self.pool1(x)
        return embedding

# ---------------------------------------------------- ResNet18 model -----------------------------------------------

class resNet18(nn.Module):
    def __init__(self):
        super(resNet18, self).__init__()
        # Load the pre-trained davit model
        resnet18_model = resnet18(weights='IMAGENET1K_V1', progress = False)
        
        # extract model except last layer
        self.features = nn.Sequential(*list(resnet18_model.children())[:-2])
        self.pool1 = nn.AvgPool2d(kernel_size=1, stride=1)

    def forward(self, x):
        x = self.features(x)
        embedding = self.pool1(x)
        # x = F.adaptive_avg_pool2d(x, (1, 1))   # To get desired dimension
        # x = x.view(x.size(0), -1)
        return embedding

# -------------------------------- Small model with Softmax layers v1 --------------------------------------------------

class ComparativeModel(nn.Module):
    def __init__(self, encoder, image_size, DEVICE):
        super(ComparativeModel, self).__init__()
        self.encoder = encoder 

        # Determine the input size for linear layers by passing a dummy input through the encoder
        dummy_input = torch.randn(1, 3, image_size, image_size).to(DEVICE)
        encoder_output = self.encoder(dummy_input).to(DEVICE)
        encoder_output_size = encoder_output.size(1)

        # Calculate mean and std of weights
        encoder_params = self.encoder.parameters()
        all_weights = torch.cat([p.data.view(-1) for p in encoder_params])
        
        trained_weights_mean = all_weights.mean().item()
        trained_weights_std = all_weights.std().item()
        

        self.conv1 = nn.Conv2d(in_channels=encoder_output_size, out_channels=32, kernel_size=2, stride=2)
    

        # Initialize the weights and bias
        init.normal_(self.conv1.weight, mean=trained_weights_mean, std=trained_weights_std)
        init.constant_(self.conv1.bias, 0.0)  # type: ignore # bias to zero -- > test
        
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(in_features=1, out_features=4)

    def forward(self, x1, x2):
        x1 = self.embedding_extraction(x1)
        
        x2 = self.embedding_extraction(x2)
        
        cosine_similarity = F.cosine_similarity(x1,x2)
        cosine_similarity = cosine_similarity.unsqueeze(1)
        
        output = F.softmax(self.linear(cosine_similarity), dim = -1)
        
        return output
    
    def to_forward_pass(self,embed):
        x = F.relu(self.conv1(embed))
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        return x
    
    def embedding_extraction(self,x):
        x = self.encoder(x)
        x = self.to_forward_pass(x)
        return x
    
# -------------------------------- Modified Small model with Softmax layers v2--------------------------------------------------

class ModifiedComparativeModel(nn.Module):
    def __init__(self, encoder, output_shape, image_size, DEVICE):
        super(ModifiedComparativeModel, self).__init__()
        self.encoder = encoder 

        # Determine the input size for linear layers by passing a dummy input through the encoder
        dummy_input = torch.randn(1, 3, image_size, image_size).to(DEVICE)
        encoder_output = self.encoder(dummy_input).to(DEVICE)
        encoder_output_size = encoder_output.size(1)
        
        self.custom_conv_layers = nn.Sequential(
            nn.Conv2d(encoder_output_size, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.custom_linear_layers = nn.Sequential(
            nn.Linear(in_features=128*3*3, out_features=32),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Linear(in_features=32, out_features = output_shape)
        )
        
        
    def forward(self, x1, x2):
        x1 = self.encoder(x1)
        x2 = self.encoder(x2)

        # concatenated = torch.cat((x1, x2), dim=1)
        concatenated = x1 - x2
        x3 = self.custom_conv_layers(concatenated)
        x4 = x3.view(x3.size(0), -1)
        x5 = self.custom_linear_layers(x4)
        b_out = F.softmax(x5, dim=1)
        return b_out