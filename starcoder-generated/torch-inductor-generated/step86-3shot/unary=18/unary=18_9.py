
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(3, 7, kernel_size=1)
        self.conv1_bn = torch.nn.BatchNorm2d(7)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.conv1_bn(v2)
        return torch.flatten(v3, 1)
# Inputs to the model
x1 = torch.randn(1, 3, 10, 10)
