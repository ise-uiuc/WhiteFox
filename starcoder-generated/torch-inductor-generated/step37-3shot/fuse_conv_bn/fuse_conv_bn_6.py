
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2)
        self.conv2 = torch.nn.Conv2d(64, 32, kernel_size=1)
        self.conv3 = torch.nn.Conv2d(32, 20, kernel_size=3, padding=1)
        self.fc = torch.nn.Linear(20 * 8 * 8, 120)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.bn2 = torch.nn.BatchNorm2d(32)
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.bn1(x1)
        x3 = self.conv2(x2)
        x4 = self.bn2(x3)
        x5 = self.conv3(x4)
        x6 = x5.view(x5.size(0), -1) # Flatten
        x7 = self.fc(x6)
        return x7
# Inputs to the model
x = torch.randn(1, 64, 8, 8)
