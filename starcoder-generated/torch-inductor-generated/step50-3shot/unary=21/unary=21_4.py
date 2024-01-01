
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(3, 11, 18, stride=2, padding=9)
        self.tanh = torch.nn.Tanh()
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.tanh(v1)
        v3 = v2.sum(dim=(-1))
        return v3
# Inputs to the model
x = torch.randn(24, 3, 64)
