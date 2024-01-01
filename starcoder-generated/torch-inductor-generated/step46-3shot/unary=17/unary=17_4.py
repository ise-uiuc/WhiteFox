
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block0 = torch.nn.Sequential(torch.nn.ConvTranspose2d(3, 16, 2, padding=(1, 1), stride=2, bias=False), torch.nn.ReLU(inplace=True), torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True))
    def forward(self, x1):
        y = self.block0(x1)
        return y
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
