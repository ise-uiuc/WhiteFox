
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=0, padding=0, dilation=0, groups=1, bias=True)
        self.relu = torch.nn.ReLU()
        self.conv_2 = torch.nn.Conv2d(3, 3, 1, stride=1, padding=0, dilation=1, groups=3, bias=True)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.relu(v1)
        v3 = self.conv_2(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 2, 2)
