
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(1, 1, 2)
        self.conv2 = torch.nn.ConvTranspose2d(1, 1, 2)
        self.bn = torch.nn.BatchNorm2d(1, running_mean=[1.0], running_var=[1.0])
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        return x
# Inputs to the model
x = torch.randn(1, 1, 4, 4)
