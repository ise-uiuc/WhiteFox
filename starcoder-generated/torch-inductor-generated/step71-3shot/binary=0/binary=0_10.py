
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 3, stride=1, padding=1)
    def forward(self, x1, bias=None, other=5):
        v1 = self.conv(x1)
        if bias == None:
            bias = torch.randn(v1.shape)
        v2 = v1 + bias
        v3 = v2 + 1
        v4 = v3 + 2
        v5 = v4 + 3
        v6 = v5 + 4
        v7 = v6 + other
        return v7
# Inputs to the model
x1 = torch.randn(1, 1, 3, 3)
