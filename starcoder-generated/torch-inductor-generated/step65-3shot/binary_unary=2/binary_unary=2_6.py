
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, stride=2, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.Flatten(),
            torch.nn.ReLU(),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, 3, stride=2, padding=1),
            torch.nn.BatchNorm2d(8),
            torch.nn.Flatten(),
            torch.nn.ReLU(),
        )
        self.fc = torch.nn.Linear(144, 120)

    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = v1 - v2
        v4 = self.fc(v3)
        #v5 = softmax(v4)
        return v4, v3
# Inputs to the model
x1 = torch.randn(1, 1, 32, 32)
