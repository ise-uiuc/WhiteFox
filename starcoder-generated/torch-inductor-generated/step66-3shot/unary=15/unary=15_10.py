
class Model(torch.nn.Module): 
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, 1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.fc = torch.nn.Linear(16 * 112 * 112, 1000, bias=True)
        self.fc2 = torch.nn.Linear(1000, 1, bias=True)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v3 = v3.permute(0, 3, 2, 1)
        v3 = torch.flatten(v3, 1)
        v4 = self.fc(v3)
        v5 = self.fc2(v4)
        return torch.sigmoid(v5)
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
