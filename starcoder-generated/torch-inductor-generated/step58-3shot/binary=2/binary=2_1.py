
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(5, 8, 2, stride=2, padding=0)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - torch.ones((56, 56), dtype=torch.float32)
        return v2
# Inputs to the model
x = torch.randn(1, 5, 64, 64)
