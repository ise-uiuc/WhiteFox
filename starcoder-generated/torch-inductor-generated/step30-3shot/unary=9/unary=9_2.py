
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(5, 8, 1, stride=1, padding=1, dilation=1)
    def forward(self, x1_X0):
        v19 = self.conv(x1_X0)
        v20 = 2 + v19
        v21 = torch.ops.aten.opset11.clamp(v20, 0, 6)
        v22 = v21 / 6
        return v22
# Inputs to the model
x1_X0 = torch.randn(1, 5, 64, 64)
