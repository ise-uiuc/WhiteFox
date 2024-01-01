
class tanhActivation(torch.nn.Module):
    def forward(self, x2):
        v7 = torch.tanh(x2)
        return v7
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 128, 5)
        self.tanh = tanhActivation()
        self.conv2 = torch.nn.Conv2d(128, 64, 3)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.tanh(v0)
        v3 = self.conv2(v2)
        v4 = self.tanh(v3)
        return v4.detach()
# Inputs to the model
x = torch.randn(3, 3, 64, 64)
