
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(8, 8, 5, stride=1, padding=0)
    def forward(self, x1):
        x1 = self.conv1(x1)
        x2 = x1 - 0.75
        x3 = F.relu(x2)
        x4 = self.conv1(x3)
        x5 = x4 - 1
        x6 = F.relu(x5)
        x7 = self.conv1(x6)
        x8 = x7 + 0.25
        x9 = F.relu(x8)
        return x9
# Inputs to the model
x1 = torch.randn(1, 8, 12, 12)
