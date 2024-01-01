
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 10, 1, stride=1, padding=1)
    def forward(self, x1, other=None):
        v1 = self.conv(x1)
        if other == None:
            other = torch.randn(v1.shape)
        if other == 1:
            v2 = v1 + other
        elif other == 2:
            v3 = v1 + other + 1
        elif other == 3:
            v4 = v1 + other + 2
        elif other == 4:
            v5 = v1 + other + 3
        elif other == 5:
            v6 = v1 + other + 4
        elif other == 6:
            v7 = v1 + other + 5
        elif other == 7:
            v8 = v1 + other + 6
        elif other == 8:
            v9 = v1 + other + 7
        elif other == 9:
            v10 = v1 + other + 8
        else:
            v11 = v1 + other + 9
        return v1
# Inputs to the model
x1 = torch.randn(2, 1, 64, 64)
other = 1
