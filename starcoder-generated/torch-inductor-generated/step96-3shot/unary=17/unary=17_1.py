
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(3, 3, kernel_size=7, stride=2, bias=False)
        self.maxpool = torch.nn.MaxPool2d(3)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.maxpool(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
