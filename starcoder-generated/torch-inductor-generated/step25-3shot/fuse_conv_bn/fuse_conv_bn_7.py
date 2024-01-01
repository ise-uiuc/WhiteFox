
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 5)
        self.bn = torch.nn.BatchNorm2d(3)
        self.conv2 = torch.nn.Conv2d(64, 128, 5)
        self.relu = torch.nn.ReLU(True)
        self.conv3 = torch.nn.Conv2d(128, 128, 7)
    def forward(self, x):
        conv1 = self.conv1(x)
        bn = self.bn(conv1)
        conv2 = self.conv2(self.relu(bn))
        conv3 = self.conv3(self.relu(conv2))
        return conv3
# Inputs to the model
x = torch.randn(1, 3, 32, 32)
