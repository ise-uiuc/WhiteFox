
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x):
        v1 = torch.relu(self.conv(x))
        v2 = v1 + x
        return v2
# Inputs to the model
x = torch.randn(1, 16, 64, 64)
