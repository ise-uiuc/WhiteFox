
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(30, 40, 2)
    def forward(self, x):
        v1 = torch.sigmoid(x)
        v2 = torch.tanh(v1)
        v3 = torch.sigmoid(v2)
        v4 = self.conv(v1)
        return torch.tanh(v3), v4
# Inputs to the model
x = torch.randn(10, 30, 100, 100)
