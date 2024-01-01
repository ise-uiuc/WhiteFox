
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 5, stride=1, padding=2)
    def forward(self, x):
        v1 = self.conv1(x)
        a1 = self.conv1(x)
        c1 = self.conv1(x)
        v2 = v1 + a1
        v3 = v2 + c1
        v4 = torch.relu(v3)
        return v4
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
