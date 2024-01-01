
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 32, 4, stride=1, padding=3)
    def forward(self, x):
        x = self.conv(x)
        x = torch.tanh(x)
        x = torch.tanh(x)
        x = torch.tanh(x)
        x = self.conv2(x)
        x = torch.tanh(x)
        x = self.conv3(x)
        x = torch.tanh(x)
        return x
# Inputs to the model
x = torch.randn(3, 3, 128, 128)
