
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(3, 7, 3, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv(x)
        return torch.tanh(v1)
# Inputs to the model
x = torch.randn(2, 3, 32)
