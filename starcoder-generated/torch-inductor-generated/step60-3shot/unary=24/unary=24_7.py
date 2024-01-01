
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 35, 1, stride=1, padding=0)
        self.max_pooling2d = torch.nn.MaxPool2d(2, 2)
    def forward(self, x):
        negative_slope = -0.231064306974411
        v1 = self.conv(x)
        v2 = v1 > 0
        v3 = v1 * negative_slope
        v4 = torch.where(v2, v1, v3)
        v5 = self.max_pooling2d(v4)
        v6 = v5 > 0
        v7 = v5 * negative_slope
        v8 = torch.where(v6, v5, v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 16, 292, 563)
