
class tanhActivation(torch.nn.Module):
    def forward(self, x0):
        result = torch.tanh(x0)
        y = torch.add(x0, result)
        return result
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
    def forward(self, x):
        v1 = torch.tanh(self.conv(x))
        v2 = tanhActivation()(v1)
        return v2
# Inputs to the model
x = torch.randn(1, 3, 5, 5)
