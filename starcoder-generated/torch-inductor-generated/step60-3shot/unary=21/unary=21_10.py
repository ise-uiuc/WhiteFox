
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv  = torch.nn.Conv2d(3, 16, 1, stride=1, padding=3)
    def forward(self, x):
        y = self.conv(x)
        return torch.tanh(y)
# Inputs to the model
x = torch.randn(1, 3, 28, 28)
