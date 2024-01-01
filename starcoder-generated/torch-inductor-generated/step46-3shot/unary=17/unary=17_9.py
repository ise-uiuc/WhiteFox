
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block0 = torch.nn.Sequential(torch.nn.ConvTranspose2d(6, 16, 2, padding=0, stride=2, bias=False), torch.nn.ReLU(inplace=False), torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False))
    def forward(self, x):
        y = self.block0(x)
        return y
# Inputs to the model
x = torch.randn(1, 6, 32, 32)
