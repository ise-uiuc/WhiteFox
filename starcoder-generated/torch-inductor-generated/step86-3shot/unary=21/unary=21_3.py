
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 3, stride=1, padding=1)
    def forward(self, x):
        y = self.conv1(x)
        n1 = torch.tanh(y)
        z = self.conv1(x)
        n2 = torch.tanh(z)
        return n2
# Inputs to the model
x = torch.randn(1, 1, 128, 128)
