
class f(torch.nn.Module):
    def forward(self, x1):
        x2 = torch.nn.functional.softplus(x1)
        x3 = torch.nn.functional.tanh(x1)
        x4 = torch.nn.functional.tanh(x2)
        result = torch.add(x1, x3, x4, x3)
        return result
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(3, 6, 3, stride=2, padding=1)
        self.conv_2 = torch.nn.Conv2d(6, 9, 3, stride=2, padding=1)
        self.conv_3 = torch.nn.Conv2d(9, 12, 3, stride=1, padding=1)
        self.conv_4 = torch.nn.Conv2d(9, 15, 3, stride=1, padding=1)
        self.f = f()
    def forward(self, x):
        v1 = self.conv_1(x)
        v2 = self.conv_2(v1)
        v3 = self.conv_3(v2)
        v4 = self.conv_4(v1)
        v5 = self.f(v3)
        v6 = self.f(v4)
        result = torch.add(v1, v5, v2, v6)
        return result
# Inputs to the model
x = torch.randn(1, 3, 256, 256)
