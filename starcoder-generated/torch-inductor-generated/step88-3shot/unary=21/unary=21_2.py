
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(32, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
        self.conv3 = torch.nn.Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
        self.conv4 = torch.nn.Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.bn3 = torch.nn.BatchNorm2d(128)
        self.bn4 = torch.nn.BatchNorm2d(128)
        self.relu = torch.nn.ReLU(inplace=True)
    def forward(self, x10):
        x0 = self.bn1(x10)
        x1 = self.relu(x0)
        x2 = self.conv1(x1)
        x3 = self.bn2(x2)
        x4 = self.relu(x3)
        x5 = self.conv2(x4)
        x6 = self.bn3(x5)
        x7 = self.relu(x6)
        x8 = self.conv3(x7)
        x9 = self.bn4(x8)
        x11 = self.relu(x9)
        x12 = self.conv4(x11)
        x13 = torch.nn.tanh()(x12)
        return x13
# Inputs to the model
x10 = torch.randn(1, 32, 256, 256)
