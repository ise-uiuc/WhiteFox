
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3)
    def forward(self, x):
        x1 = self.conv(x)
        x = torch.tanh(x1)
        x = torch.tanh(self.conv(x))
        x = torch.sigmoid(self.conv(x))
        x = torch.tanh(self.conv(x))
        x = self.conv(x)
        return x
# Inputs to the model
x = torch.randn(1, 3, 128, 128)
