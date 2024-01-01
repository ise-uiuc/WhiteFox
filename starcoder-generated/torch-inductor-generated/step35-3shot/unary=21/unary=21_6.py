
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tanh = torch.nn.Tanh()
        self.conv = torch.nn.Conv2d(64, 64, 3, padding=1)
    def forward(self, x):
        v1 = self.conv(x)
        # v2 = torch.relu(v1)
        v2 = self.tanh(v1)
        return v2
# Inputs to the model
x = torch.randn(1, 64, 10, 10)
