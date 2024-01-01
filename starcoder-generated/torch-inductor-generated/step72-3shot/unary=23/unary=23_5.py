
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pooling = torch.nn.MaxPool2d(kernel_size=31)
        self.conv_transpose = torch.nn.ConvTranspose2d(2048, 1024, 1, groups=2048, bias=True)
    def forward(self, image):
        x1 = self.pooling(image)
        v1 = self.conv_transpose(x1)
        v2 = torch.relu(v1)
        return v2
# Inputs to the model
import torch
image = torch.randn(1, 2048, 8, 8)
