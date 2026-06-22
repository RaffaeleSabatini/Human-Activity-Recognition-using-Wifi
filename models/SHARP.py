import torch
from torch import nn

class SHARP(nn.Module):
    '''
        Simplified version of Inception-v4 Neural Network
    '''
    def __init__(self, n_features):
        super().__init__()

        self.branch1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=5, kernel_size=2, stride=2),
            nn.ReLU()
            )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=6, out_channels=9, kernel_size=4, stride=2, padding=2),
            nn.ReLU()
        )
        self.out = nn.Sequential(
            nn.Conv2d(in_channels=15, out_channels=3, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=25500, out_features=n_features) # corresponds to number of activities
        )
    
    def forward(self, x, return_pred):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x = torch.cat([x1, x2, x3], dim=1)
        y = self.out(x)
        
        if return_pred:
            return torch.argmax(y), y
        else:
            return y
