
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 4, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.nn.functional.relu(v1)
        v3 = self.conv1(v2)
        v4 = torch.nn.functional.relu(v3)
        v5 = self.conv1(v4)
        v6 = torch.nn.functional.relu(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
