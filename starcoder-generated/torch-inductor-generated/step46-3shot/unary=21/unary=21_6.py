
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 32, (3, 3), stride=(3,3))
    def forward(self, x):
        x1 = self._tanh(self.conv(x))
        return x1
def _tanh(x):
        return torch.tanh(x)
def _sigmoid(x):
        return torch.sigmoid(x)
# Inputs to the model
x = torch.randn(1, 3, 64, 64, requires_grad=True)
