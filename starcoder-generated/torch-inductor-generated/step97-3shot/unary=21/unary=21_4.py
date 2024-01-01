
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(4, 2, 3, stride=1)
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = torch.tanh(x1)
        return torch.tanh(x2)
# Inputs to the model
x = torch.randn(1, 4, 64, 64)
