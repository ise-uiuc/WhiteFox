
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.conv2 = torch.nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.conv3 = torch.nn.Conv2d(64, 1, 1)
    def forward(self, x):
        # v1 = self.conv1(x)
        out = self.conv2(x)
        out = self.conv3(out)
        return out
# Inputs to the model
x1 = torch.randn(1, 3, 256, 64)
# Input to the model
