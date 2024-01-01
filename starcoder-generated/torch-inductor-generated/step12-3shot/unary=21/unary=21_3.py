
class conv2d_tanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, (1,), (1,), 0)
    def forward(self, x):
        tanh = torch.tanh(self.conv(x))
        return tanh
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m1 = conv2d_tanh()
    def forward(self, x):
            v1 = self.m1(x)
            v2 = torch.tanh(v1)
            return v2
# Inputs to the model
tensor = torch.randn((1, 3, 128, 128))
