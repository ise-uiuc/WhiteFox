
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 7, 1)
        self.tanh = torch.nn.Tanh()
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.tanh(v1)
        return v2
# Inputs to the model
x = torch.randn(1, 1, 28, 28)
