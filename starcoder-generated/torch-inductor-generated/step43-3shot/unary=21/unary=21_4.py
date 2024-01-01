
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose1d(3, 10, 3, stride=1)
        self.conv2 = torch.nn.ConvTranspose2d(10, 3, 3, stride=1)
    def forward(self, x):
        v2 = self.conv1(x)
        v3 = torch.tanh(v2)
        v4 = self.conv2(v3)
        return torch.tanh(v4)
# Inputs to the model
x = torch.randn(1, 3, 1, 1)
