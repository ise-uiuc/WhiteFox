
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(5, 5, 1)
        self.conv2 = torch.nn.Conv2d(5, 5, 1)
        self.conv3 = torch.nn.Conv2d(5, 1, 1)
    def forward(self, x):
        v = self.conv1(x)
        v = torch.tanh(v)
        v = self.conv2(v)
        v = torch.tanh(v)
        v = self.conv3(v)
        v = torch.tanh(v)
        return v
# Inputs to the model
v = torch.randn(1, 5, 32, 32)
