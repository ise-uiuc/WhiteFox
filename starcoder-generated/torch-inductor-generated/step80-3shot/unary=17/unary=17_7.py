
import torch.nn as nn
class Model(nn.Module):
    def __init__(self, n_in, n_1, n_2, n_3):
        super(Model, self).__init__()
        self.conv1 = nn.ConvTranspose2d(n_in, n_1, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(n_1, n_2, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.ConvTranspose2d(n_2, n_3, kernel_size=5, stride=1, padding=2)   
    def forward(self, x):
        x = self.conv1(x) # size=(1, n_1, x.size(2) * 2, x.size(3) * 2)
        x = self.conv2(F.relu(x))
        x = self.conv3(F.relu(x))
        return x
# Inputs to the model
x = torch.randn(1, 5, 10, 10)
