
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3, stride=2, padding=1)
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        v3 = self.conv(x1)
        v1 = v3 - 0.0003
        v2 = self.relu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 75, 75)
