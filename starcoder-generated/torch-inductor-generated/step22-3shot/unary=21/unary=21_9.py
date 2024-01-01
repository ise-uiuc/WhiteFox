
class tanhActivation(torch.nn.Module):
    def forward(self, x310):
        result = torch.tanh(x310)
        y = torch.add(x310, result)
        return result
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 8, 3, stride=2, padding=2)
        self.tanh = tanhActivation()
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.tanh(v1)
        return v2
# Inputs to the model
x = torch.randn(64, 1, 64, 64)
