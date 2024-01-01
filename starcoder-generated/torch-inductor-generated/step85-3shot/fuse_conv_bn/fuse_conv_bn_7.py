
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32, affine=False)
        self.conv2 = torch.nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(32, affine=False)
        self.conv3 = torch.nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(64, affine=False)
        self.conv4 = torch.nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn4 = torch.nn.BatchNorm2d(64, affine=False)
        self.relu = torch.nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        return x
# Inputs to the model
x = torch.randn(1, 3, 224, 224)
