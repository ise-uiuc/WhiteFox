
import torch
class Model(torch.nn.Module):
        def __init__(self):
            super(self.__class__, self).__init__()

            self.features = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(6, 10, (13, 13), stride = (1, 1), padding = (1, 1)),
            )
        def forward(self, x):
            f = self.features(x)
            return f
# Inputs to the model
x = torch.randn(1, 6, 32, 32)
