
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block0 = torch.nn.Sequential(torch.nn.ConvTranspose2d(3, 16, 2, padding=1, stride=2, bias=False), torch.nn.ReLU(inplace=False), torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False), torch.nn.ConvTranspose2d(16, 16, 2, padding=1, stride=2, bias=False), torch.nn.ReLU(inplace=False), torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False), torch.nn.ConvTranspose2d(16, 16, 2, padding=1, stride=2, bias=False), torch.nn.ReLU(inplace=False))
    def forward(self, x1):
        y = self.block0(x1)
        return torch.flatten(y, 1)
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
