
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(8, 1, 5, stride=1, padding=2)
        self.pool = torch.nn.AdaptiveAvgPool2d((14, 14))
    def forward(self, x):
        negative_slope = 1
        v1 = self.conv1(x)
        v2 = v1 > 0
        v3 = v1 * negative_slope
        v4 = torch.where(v2, v1, v3)
        v5 = self.pool(v4)
        v6 = v5.flatten(1)
        return v6
# Inputs to the model
x1 = torch.randint(low=-1, high=2, size=(1, 8, 117, 117), dtype=torch.float32, requires_grad=True)
