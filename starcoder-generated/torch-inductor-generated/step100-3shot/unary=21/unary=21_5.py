
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(12, 13, 2, stride=2)
        self.conv2 = torch.nn.Conv2d(11, 14, 1, stride=1, padding=1)
    def forward(self, x):
        x = self.conv(x)
        x = torch.tanh(x)
        return self.conv2(x)
# Inputs to the model
x = torch.randn(1, 12, 32, 32)
