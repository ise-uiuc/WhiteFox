
class ModelTanh(nn.Module):
    def __init__(self):
        super(ModelTanh, self).__init__()
        self.conv = nn.Conv2d(1, 1, 11, border_mode=0, dilation=2, groups=1, bias=False, stride=1)
    def forward(self, x):
        y = self.conv(x)
        z = torch.tanh(y)
        return z
# Inputs to the model
x = torch.rand(2, 1, 256, 128)
