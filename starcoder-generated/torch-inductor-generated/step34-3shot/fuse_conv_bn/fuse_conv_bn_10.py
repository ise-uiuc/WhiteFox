
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv3d(3, 3, 4)
        self.conv2 = torch.nn.Conv3d(3, 3, 3)
        self.conv3 = torch.nn.Conv3d(3, 3, 2)
        self.batchnorm3d = torch.nn.BatchNorm3d(3)
    def forward(self, x1):
        s = self.conv1(x1)
        t = self.conv2(s)
        t = self.conv3(t)
        t = self.batchnorm3d(t)
        return t
# Inputs to the model
x1 = torch.randn(1, 3, 6, 6, 6)
