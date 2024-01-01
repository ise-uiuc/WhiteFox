
class tanhActivation(torch.nn.Module):
    def forward(self, x):
        return torch.tanh(x)
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 3, 1)
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = tanhActivation()
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.sigmoid(v1)
        v3 = self.tanh(v2)
        return v3
# Input to the model
x = torch.randn(20, 1, 10, 10)
