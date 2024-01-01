
class ModelA(torch.nn.Module):
    def __init__(self, d=3, is_fuse=False):
        super(ModelA, self).__init__()
        self.conv1 = torch.nn.Conv2d(d, d, 5, 1, 1)
        self.bn1 = torch.nn.BatchNorm2d(d)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(d, d, 3, 1, 1)
        self.bn2 = torch.nn.BatchNorm2d(d)
        self.is_fuse = is_fuse

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        if self.is_fuse:
            x = self.conv2(x)
            return x
        else:
            x = self.bn2(self.conv2(x))
            return x
# Inputs to the model
x = torch.randn(1, 3, 12, 12)
