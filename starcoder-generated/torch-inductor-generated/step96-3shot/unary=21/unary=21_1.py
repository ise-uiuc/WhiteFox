
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(11, 20, 3, padding=3)
        self.conv2 = torch.nn.Conv2d(33, 64, 3, bias=False)
        self.conv3 = torch.nn.Conv2d(65, 128, 3)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.tanh(v1)
        v3 = v2
        v3 = v2 + v3
        v4 = v3 + self.conv2(v2)
        v5 = torch.tanh(v4)
        v5 = self.conv3(v5)
        return v5
# Inputs to the model
x = torch.randn(1, 11, 32, 32)
