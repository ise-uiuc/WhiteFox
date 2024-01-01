
class ModelM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_A1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv1_B1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv1_C1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1, x2):
        v1a1 = self.conv1_A1(x1)
        v1b1 = self.conv1_B1(x1)
        v1c1 = self.conv1_C1(x1)
        v1 = v1a1 + v1b1 + v1c1
        v2 = v1 + x2
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
