
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 1, stride=1, padding=1)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.relu(v1)
        v3 = v2 - True
        return v3
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
