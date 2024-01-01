
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(9)
        self.conv = torch.nn.MaxPool2d(1)
        self.tanh = torch.nn.Tanh()
    def forward(self, x):
        t1 = self.bn(x)
        t2 = self.conv(t1)
        t3 = self.tanh(t2)
        return t3
# Inputs to the model
tensor = torch.randn(1, 9, 16, 16)
