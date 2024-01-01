
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1, padding=3, bias=False)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - 1
        v3 = F.relu6(v2)
        return v3
# Inputs to the model
x = torch.randn(1, 1, 28, 28)
