
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(20, 50, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(50, 10, 3, stride=1, padding=1)
    def forward(self, x):
        x = torch.tanh(self.conv(x))
        x = self.conv2(x)
        return x
# Inputs to the model
x = torch.randn(1, 20, 28, 28)
