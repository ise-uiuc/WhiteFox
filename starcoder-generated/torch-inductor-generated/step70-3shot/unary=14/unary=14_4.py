
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(19, 33, 6, stride=2, padding=3, dilation=1, groups=1, bias=False, padding_mode='zeros')
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.relu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 19, 192, 192)
