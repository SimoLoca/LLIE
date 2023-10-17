import torch
import torch.nn as nn


class DCE_Net(nn.Module):
    def __init__(self, n_layers: int = 8):
        super().__init__()
        self.n = n_layers
        # RGB input image
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        # Residual-like layers 
        self.conv5 = nn.Conv2d(32*2, 32, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(32*2, 32, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(32*2, 24, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

    
    def forward(self, x):
        out1 = self.relu(self.conv1(x))
        out2 = self.relu(self.conv2(out1))
        out3 = self.relu(self.conv3(out2))
        out4 = self.relu(self.conv4(out3))
        out5 = self.relu(self.conv5(torch.hstack((out4, out3))))
        out6 = self.relu(self.conv6(torch.hstack((out5, out2))))
        x_r = self.tanh(self.conv7(torch.hstack((out6, out1))))

        r = torch.split(x_r, 3, dim=1)
        
        enhance_image = x
        # Produces 24 parameter maps for 8 iterations, where each iteration 
        # requires 3 curve parameter maps for the three channels
        for i in range(self.n):
            enhance_image = enhance_image + r[i] * (torch.pow(enhance_image, 2) - enhance_image)
        
        return enhance_image, torch.hstack(*[r])
    
