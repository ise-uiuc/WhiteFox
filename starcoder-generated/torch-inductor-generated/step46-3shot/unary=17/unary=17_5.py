
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block0 = torch.nn.Sequential(torch.nn.ConvTranspose2d(3, 16, 3, padding=1, stride=2, bias=False, output_padding=1),torch.nn.BatchNorm2d(16),torch.nn.ReLU(inplace=True),torch.nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=0, dilation=1, ceil_mode=False))
    def forward(self, x1):
        y = self.block0(x1)
        return y
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
