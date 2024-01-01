
class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7)
        self.norm1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        self.fc = nn.Linear(2048, 256)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(self.norm1(x), 3, stride=2)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 3, stride=2)
        x = x.view(-1, 2048)
        x = F.relu(self.fc(x))
        return x
# Inputs to the model
x = torch.randn(1, 3, 224, 224)
