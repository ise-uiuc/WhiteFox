
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(3, 16, 3, stride=2, padding=1)
        self.conv_2 = torch.nn.Conv2d(16, 24, 3, stride=2, padding=1)
    def forward(self, x1, x2, conv_2=None):
        v1 = self.conv_2(self.conv_1(x1))
        if conv_2 is None:
            conv_2 = torch.randn(v1.shape)
        v2 = v1 + conv_2 
        v3 = x2 + v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 32, 32)
