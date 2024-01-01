
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(128, 64, 1, stride=2, padding=1)
        self.upsample = torch.nn.Upsample(scale_factor=4.0, mode='nearest', align_corners=True)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.upsample(v1)
        v3 = v2 * 0.5
        v4 = v2 * v2
        v5 = v4 * v2
        v6 = v5 * 0.044715
        v7 = v2 + v6
        v8 = v7 * 0.7978845608028654
        v9 = torch.tanh(v8)
        v10 = v9 + 1
        v11 = v3 * v10
        return v11
# Inputs to the model
x1 = torch.randn(1, 128, 4, 5)
