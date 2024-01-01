
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 4, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1.add(3)
        v3 = v2.clamp(0, 6)

        # The following operations perform ReLU6 or its gradient (ReLU6 gradient).
        v4 = v3.max(torch.zeros_like(v3))
        v5 = v4.min(((torch.full_like(v4, 6, dtype=torch.float64))).item())
        v6 = v5.div(6)

        v4 = v3.masked_fill(v3>6, -1e3)
        v5 = v4.masked_fill(v4<0, 6)
        v6 = v5/6

        # The following operations perform ReLU6 or its gradient (ReLU6 gradient).
        v7 = v3.where(v3 < 6, torch.ones_like(v3)*6)
        v8 = v7 / 6

        return self.conv2(v8)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
