
class Model(torch.nn.Module):
    def __init__(self, N):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 100, 1)
        self.bn = torch.nn.BatchNorm2d(100)
        for _ in range(1, N):
            self.conv += torch.nn.Conv2d(100, 100, 1)
        self.fc = torch.nn.Linear(105 * 137, 1000)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x).relu()
        # Flatten, i.e. change from matrix to vector
        return self.fc(x.view(1, -1))
# Inputs to the model
x = torch.randn(1, 1, 128, 127)
