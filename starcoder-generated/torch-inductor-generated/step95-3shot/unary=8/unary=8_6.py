
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 1, 3, stride=1)
        self.avgpool = torch.nn.AvgPool2d(3, stride=1, padding=1)
        self.t1 = torch.randn(1, 1, 3, 3)
        self.tanh = torch.nn.Tanh()
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        y1 = self.avgpool(v1)
        v2 = self.t1 / 9.0
        z1 = torch.clamp(v2, min=0)
        y2 = self.tanh(y1)
        v3 = self.t1 / 9.0
        r1 = torch.clamp(v3, min=0)
        v4 = z1 * r1
        v5 = y2 * v4
        v6 = v5 / 9.0
        return v6
# Inputs to the model
x1 = torch.randn(1, 1, 3, 3)
