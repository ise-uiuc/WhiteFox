
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = torch.nn.MaxPool2d(2, stride=1, padding=2)
    def forward(self, x1):
        v1 = self.maxpool(x1)
        v2 = math.add(v1, 3.0)
        v3 = math.clamp(v2, 0.0, 6.0)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 15, 15)
