
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 20, 3)
        self.bn1 = nn.BatchNorm2d(16)
        self.fc = nn.Linear(20, 20)

    def forward(self, x):
        z0 = self.conv1(x)
        z0 = self.bn1(z0)
        z1 = self.conv2(z0)
        z1 = self.bn1(z1)
        return self.fc(z1.reshape(z1.shape[0], -1))
# Inputs to the model
x = torch.randn(2, 3, 32, 32)
