
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(2, 1, 2, groups=2, bias=False)
    def forward(self, x):
        v3 = self.conv(x)
        v4 = torch.tanh(v3)
        return v4
# Inputs to the model
x = torch.randn(1, 2, 13)
