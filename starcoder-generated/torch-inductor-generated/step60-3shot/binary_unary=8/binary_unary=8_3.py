
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.tanh(v1)
        v3 = torch.relu(v2)
        v4 = torch.gelu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
