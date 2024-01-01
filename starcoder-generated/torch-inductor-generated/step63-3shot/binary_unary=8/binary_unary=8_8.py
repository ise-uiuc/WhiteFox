
# Leverages an unsupported function
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 4, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv(x1)
        v3 = v1.tanh()
        v4 = v3 + v2
        # v5 = torch.relu(v4)
        return v4
# Inputs to the model
x1 = torch.randn(1, 2, 64, 64)
