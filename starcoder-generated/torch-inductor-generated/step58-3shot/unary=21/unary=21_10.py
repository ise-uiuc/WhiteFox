
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(49, 49, 2, 1)
        self.conv2 = torch.nn.Conv2d(49, 49, 2, 1, 0, 1, 1, 1, False)
        self.conv3 = torch.nn.Conv2d(49, 7, 1, 1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.tanh(v1)
        v3 = self.conv2(v1)
        v4 = torch.tanh(v3)
        v5 = self.conv3(v4)
        return v5
# Inputs to the model
x = torch.randn(33, 49, 19, 6)
