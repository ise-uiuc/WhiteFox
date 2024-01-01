
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 64, 3, stride=2, padding=1)
        self.linear = torch.nn.Linear(64, 1)
    def forward(self, x1):
        v1 = self.conv(x1).flatten(1)
        v2 = self.linear(v1)
        v3 = (v2 * x1).sigmoid()
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
