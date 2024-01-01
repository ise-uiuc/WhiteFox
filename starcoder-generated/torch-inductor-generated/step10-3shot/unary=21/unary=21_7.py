
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(3, 3, 9, padding=4)
        self.bn = torch.nn.BatchNorm1d(3)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.bn(v1)
        return torch.tanh(v2)
# Inputs to the model
x = torch.randn(1, 3, 128)
