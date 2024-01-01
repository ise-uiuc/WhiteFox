
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 2, (1, 2), stride=2, padding=2)
        self.conv1 = torch.nn.Conv2d(2, 2, (1, 1), stride=1, padding=1)
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv1(v1)
        v3 = self.relu(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 8, 8)
