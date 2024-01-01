
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm1 = torch.nn.BatchNorm2d(64)
        self.conv2 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm2 = torch.nn.BatchNorm2d(64)
    def forward(self, x):
        x = self.batchnorm1(self.conv1(x))
        x = self.conv2(x)
        x = self.batchnorm2(x)
        return x
# Inputs to the model
x = torch.randn(1, 64, 10, 10)
