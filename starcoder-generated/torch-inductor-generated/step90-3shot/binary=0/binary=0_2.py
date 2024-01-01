
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(10, 15, 1, stride=1, padding=1)
        self.conv_2 = torch.nn.Conv2d(15, 25, 1, stride=1, padding=1)
    def forward(self, x1, other=None):
        v1 = self.conv_1(x1)
        if other == None:
            other = torch.randn(v1.shape)
        v2 = self.conv_2(v1)
        v3 = v2 + other
        return v3
# Inputs to the model
x1 = torch.randn(1, 10, 16, 16)
