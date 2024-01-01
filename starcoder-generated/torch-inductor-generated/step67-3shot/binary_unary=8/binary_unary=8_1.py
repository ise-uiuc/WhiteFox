
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv23 = torch.nn.Conv2d(23, 64, 3, stride=2, padding=1)
        self.conv45 = torch.nn.Conv2d(45, 128, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv23(x1)
        v2 = self.conv45(x1)
        v3 = torch.relu(v1 + v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 23, 32, 128)
