
class Model(torch.nn.Module):
    def __init__(self, stride=2):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 8, 3, stride=stride, padding=1)
    def forward(self, x):
        x = self.conv(x)
        x = torch.tanh(x)
        return x
# Inputs to the model
x = torch.randn(1, 1, 64, 64)
