
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 3)
        self.bn = torch.nn.BatchNorm2d(num_features=1)
    def forward(self, x2):
        x2 = torch.round(self.conv(x2))
        x2 = self.conv(x2)
        x2 = torch.tanh(x2)
        x2 = self.bn(x2)
        return torch.add(x2, x2)
# Inputs to the model
x2 = torch.randn(1, 1, 4, 4)
