
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 1, stride=1, padding=1)
    def forward(self, x):
        x = self.conv(x)
        x = torch.tanh(x)
        x = self.conv(x)
        return x
# Inputs to the model
x = torch.zeros((1, 3, 32, 32))
