
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2196, 24, 1, 1)
        self.conv2 = torch.nn.ConvTranspose2d(24, 16, 1, 1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.tanh(v1)
        v3 = self.conv2(v2)
        return v3
# Inputs to the model
x = torch.randn(614, 2196, 19, 8)
