
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 16, (1, 2), (2, 1), (1, 2))
        self.conv2 = torch.nn.Conv2d(16, 16, (2, 1), padding=(3, 1), dilation=(2, 1))
    def forward(self, x):
        v1 = torch.add(self.conv(x), self.conv2(self.conv(x)))
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
input = torch.randn(4, 2, 10, 20)
