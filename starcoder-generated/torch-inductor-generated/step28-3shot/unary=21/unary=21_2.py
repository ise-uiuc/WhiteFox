
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(3, 1, 5, stride=2)
    def forward(self, x):
        a = self.conv(x)
        b = torch.tanh(a)
        return b
# Inputs to the model
x = torch.randn(1, 3, 5)
