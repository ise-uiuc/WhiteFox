
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 1, stride=1, padding=1)

    def forward(self, x1):
        x1 = nn.ReLU(inplace=True)(x1)
        v1 = self.conv(x1)
        out1 = v1 + 3
        out2 = out1.clamp(min=0,max=6)
        out3 = out2.div(6)
        return out3
# Inputs to the model
x1 = torch.randn(5, 3, 64, 64)
