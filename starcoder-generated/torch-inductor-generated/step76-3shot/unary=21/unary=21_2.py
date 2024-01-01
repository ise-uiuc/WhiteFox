
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 64, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(64, 32, 5, stride=1, padding=2)
        self.conv3 = torch.nn.Conv2d(32, 3, 3, stride=1, padding=1)
    def forward(self, x):
        x = torch.tanh(self.conv(x))
        x = torch.tanh(self.conv2(x))
        x = torch.tanh(self.conv3(x))
        return x.sum(dim=3).view(-1)
# Inputs to the model
x = torch.randn(1, 1, 28, 28)
