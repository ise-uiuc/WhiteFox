
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x):
        v2 = self.conv1(x)
        v = v2 + x
        v1 = torch.relu(v2)
        v3 = v1 + v
        return v3
# Inputs to the model
x = torch.randn(2, 16, 64, 64)
