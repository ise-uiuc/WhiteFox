
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 100, 1, stride=1, in_channels=3, groups=5, kernel_size=[3, 3])
        self.tanh = torch.nn.Tanh()
    def forward(self, x2):
        v1 = self.conv(x2)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x2 = torch.randn(507, 3, 4, 4)
