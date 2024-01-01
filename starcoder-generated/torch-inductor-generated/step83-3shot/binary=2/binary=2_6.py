
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, (2, 4), stride=1, padding=0)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - 0.2838841
        return v2
# Inputs to the model
x = torch.randn(1, 3, 32, 32)
