
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block0 = torch.nn.Sequential(torch.nn.ConvTranspose2d(3, 16, 2, padding=1, stride=2), torch.nn.ReLU(inplace=False), torch.nn.MaxPool2d(kernel_size=3, ceil_mode=False, padding=2, dilations=1, stride=1))
    def forward(self, x):
        v = self.block0(x)
        return v
# Inputs to the model
x = torch.randn(1, 3, 16, 16)
