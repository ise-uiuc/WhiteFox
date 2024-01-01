
class ModelTanh1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(3, 7, 1, stride=1, bias=False)
    def forward(self, x2):
        v1 = self.conv(x2)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x2 = torch.randn(64, 3, 64)
