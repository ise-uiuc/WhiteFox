
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 60, 3, stride=2, padding=1)
        self.tanh = torch.nn.Tanh()
    def forward(self, x):
        v1 = torch.tanh(self.conv(x))
        # v1 = self.tanh(self.conv(x))
        v2 = torch.sigmoid(v1)
        # v2 = self.tanh(v1)
        return v2
# Inputs to the model
x = torch.randn(10, 3, 32, 32)
