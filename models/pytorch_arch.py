# app/models/pytorch_arch.py
import torch
import torch.nn as nn

class TabularNet(nn.Module):
    def __init__(self, input_dim=30, n_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,n_classes),
            #Ya no vamos a usar la funcion sigmoide si no que vamos a usar CrossEntropyLoss
            #nn.Sigmoid()
        )
    def forward(self,x): return self.net(x)


#Estas clases de abajo no las vamos a usar, pero las dejo por si acaso
class ImageCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Flatten(),
            nn.Linear(64*8*8,64), nn.ReLU(),
            nn.Linear(64,10)
        )
    def forward(self,x): return self.net(x)

class AudioCNN(nn.Module):
    def __init__(self, n_classes=35):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,32,3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(64, n_classes)
        )
    def forward(self,x): return self.net(x)
