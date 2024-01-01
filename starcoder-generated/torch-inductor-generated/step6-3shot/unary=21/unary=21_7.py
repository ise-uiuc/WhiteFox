
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(64, 64, 1, stride=1)
        self.conv2 = torch.nn.Conv2d(64, 64, 1, stride=1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.tanh(v1)
        return self.conv2(v2)
# Inputs to the model
x = torch.randn(64, 64, 128, 128)
