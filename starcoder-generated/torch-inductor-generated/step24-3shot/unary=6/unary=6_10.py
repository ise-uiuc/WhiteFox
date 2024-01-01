
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.max = torch.nn.MaxPool2d(3, stride=1, padding=2)
    def forward(self, x1):
        v1 = self.max(x1) + 3
        v2 = torch.clamp(v1, min=0, max=6)
        v3 = torch.mul(v2, 4.0)
        v4 = v3 / 5
        return v4
# Inputs to the model
x1 = torch.randn(6, 3, 14, 14)
