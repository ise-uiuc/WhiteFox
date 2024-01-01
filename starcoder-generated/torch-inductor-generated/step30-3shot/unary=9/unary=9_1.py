
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(6, 8, 3, padding=1)
        self.conv_2 = torch.nn.Conv2d(8, 4, 1)
    def forward(self, x1):
        v1 = self.conv_1(x1)
        v2 = torch.add(v1, 3)
        v3 = torch.clamp(v2, min=0, max=3)
        v4 = torch.div(v3, 3)
        v5 = self.conv_2(v4)
        v6 = torch.add(v5, 3)
        v7 = torch.clamp(v6, min=0, max=6)
        v8 = v7.div(6)
        return v8
# Inputs to the model
x1 = torch.randn(1, 6, 64, 64)
