
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 32, 5, padding=4, dilation=3)
        self.fc = torch.nn.Linear(448, 10)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - 2.1
        v3 = F.relu(v2)
        v4 = torch.flatten(v3, 1)
        v5 = self.fc(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
