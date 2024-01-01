
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(9, 3, 9, padding=4)
        self.bn = torch.nn.BatchNorm2d(3)
        self.tanh = torch.nn.Tanh()
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.bn(v1)
        return self.tanh(v2)
# Inputs to the model
x = torch.randn(1, 9, 192, 192)
