
import torch

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # torch.nn.PReLU(num_parameters=1, init=0.25, num_channels=2) in Caffe2 style
        self.layers = torch.nn.PReLU(num_parameters=1, init=0.25)
    def forward(self, x):
        x = self.layers(x)
        x = torch.stack((x, x), dim = 1)
        (x_1, x_2) = torch.chunk(x, 2, dim=1)
        x = torch.cat((x_1, x_2), dim=1)
        return x
# Inputs to the model
x = torch.randn(15, 6, 2, 2, 2)
