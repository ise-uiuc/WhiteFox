
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sub_conv = torch.nn.Conv2d(3, 8, 1, stride=2, padding=1)
        self.add_conv = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.sub_conv(x1)
        v2 = torch.add(v1, 3)
        v3 = torch.clamp(v2, min=0, max=6)
        v4 = torch.div(v3, 6)
        v5 = self.add_conv(v4)
        v6 = torch.sub(v5, 3)
        v7 = torch.clamp(v6, min=0.0, max=6.0)
        v8 = torch.div(v7, 6)
        return v8
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
