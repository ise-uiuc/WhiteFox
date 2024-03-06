
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = 3 + v1
        v3 = torch.clamp(v2, 0, 6)
        v4 = torch.randn([1, 3, v3.shape[-2], v3.shape[-1]]) * v3
        v5 = v4.div(6)
        return v5.unsqueeze(-1)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)