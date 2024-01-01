
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.ConvTranspose1d(1, 1, 3, 1, 1)
        self.conv1 = torch.nn.ConvTranspose1d(1, 1, 5, 1, 0)
        self.conv2 = torch.nn.ConvTranspose1d(1, 1, 2, 1, 2)
    def forward(self, x1):
        v1 = self.conv0(x1)
        v2 = torch.tanh(v1)
        v3 = self.conv1(v2)
        v4 = torch.tanh(v3)
        v5 = self.conv2(v4)
        v6 = torch.tanh(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 1, 10)
