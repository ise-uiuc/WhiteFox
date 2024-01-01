
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 256, (4, 4), groups=32)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v3 = torch.relu(v1)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 12, 12)
