
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 16, 7, stride=7, padding=7)
        self.activation = torch.nn.ReLU()
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.activation(v1)
        v3 = self.conv(x1)
        v4 = self.activation(v3)
        v5 = v4 + v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
