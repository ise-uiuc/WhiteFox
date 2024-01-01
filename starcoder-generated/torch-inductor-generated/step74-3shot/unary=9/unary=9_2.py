
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Conv2d = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.Conv2d(x1)
        v2 = torch.add(v1, 3, alpha=1)
        v3 = torch.clamp(v2, min=0, max=6)
        v4 = torch.div(v3, 6)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
