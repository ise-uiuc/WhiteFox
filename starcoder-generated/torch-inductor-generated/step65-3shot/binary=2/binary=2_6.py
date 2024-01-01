
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 8, 5, stride=1, padding=0)
        self.fc1 = torch.nn.Linear(2048, 1024)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - 4.0
        v3 = self.fc1(x)
        v4 = v2 + v3
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 160, 170)
