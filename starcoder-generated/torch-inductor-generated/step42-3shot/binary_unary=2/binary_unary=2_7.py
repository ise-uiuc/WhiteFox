
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 16, 1, padding=0, stride=1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - 1.0
        v3 = torch.relu(v2)
        return v3
# Inputs to the model
x = torch.randn(1, 16, 52, 52)
