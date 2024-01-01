
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(3, 16, 3, stride=1, padding=1)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.relu(v1)
        v3 = v2 - 1.0
        return v3
# Inputs to the model
x = torch.randn(1, 3, 8, 8, 8)
