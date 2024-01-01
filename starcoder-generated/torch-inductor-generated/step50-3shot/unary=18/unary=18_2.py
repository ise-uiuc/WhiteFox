
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(2)
        self.linear1 = nn.Linear(32 * 4 * 4, 16)
        self.linear2 = nn.Linear(16, 128)
        self.linear3 = nn.Linear(128, 2)
    def forward(self, x):
        x = F.relu_(self.conv1(x))
        x = F.relu_(self.conv2(x))
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        x = F.relu_(self.linear1(x))
        x = F.relu_(self.linear2(x))
        x = self.linear3(x)
        x = F.softmax(x, dim=0)
        return x
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
