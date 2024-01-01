
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(1, 4, 1, stride=1, bias=False)
    def forward(self, x):
        x = torch.tanh(self.conv(x))
        return x
# Inputs to the model
x = torch.randn(1, 1, 10)
