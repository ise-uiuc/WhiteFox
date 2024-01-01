
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - 0.0
        v3 = torch.nn.functional.relu(v2)
        return v3
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
