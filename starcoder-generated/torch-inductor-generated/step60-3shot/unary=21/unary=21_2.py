
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, stride=2)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, stride=2)
    def forward(self, x):
        t = torch.tanh(self.conv1(x))
        y = self.conv2(t)
        return torch.tanh(y)
# Inputs to the model
x = torch.randn(1, 3, 128, 128)
