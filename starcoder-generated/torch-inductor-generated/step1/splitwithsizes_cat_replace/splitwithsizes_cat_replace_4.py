
import torch
import torch.nn.functional as F

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.batch_split_size = 1
        self.channels_split_size = [2, 4]
        self.softmax = torch.nn.Softmax(dim=0)

    def forward(self, x):
        x, _ = torch.split(x, [self.batch_split_size, int(x.size(1) * sum(self.channels_split_size) / x.size(1))])
        x, _ = torch.split(x, self.channels_split_size, dim=2)
        x = self.softmax(x / 2)
        x = torch.cat((x, x, x), dim=2)
        x = x + x
        return x

# Initializing the model
device = "cpu"
num_images = 1
input_channels = 6
h = 100
w = 100
inputs = torch.randn(num_images, input_channels, h, w).to(device)
m = Model().to(device)

# Inputs to the model
x = inputs
