
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 7, stride=1, padding=3, dilation=1, bias=True, groups=1)
        self.conv2 = torch.nn.Conv2d(64, 128, 5, stride=1, padding=2, dilation=1, bias=False, groups=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = torch.tanh(v1)
        v3 = self.conv2(v2)
        v4 = torch.tanh(v3)
        return v4
# Inputs to the model
x = torch.rand(128, 3, 223, 223)
