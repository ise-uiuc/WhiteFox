
class tanhActivation(torch.nn.Module):
    def forward(self, x):
        result = torch.tanh(x)
        return result
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 3, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 1, 3, stride=1, padding=1)
        self.tanh = tanhActivation()
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = torch.tanh(v1)
        v3 = self.conv2(v2)
        t1 = torch.tanh(v3)
        return t1.detach()
# Inputs to the model
x = torch.randn(1, 1, 64, 64)
