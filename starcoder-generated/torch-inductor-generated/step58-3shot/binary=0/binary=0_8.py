
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(64, 64, 1, stride=1, padding=0)
    def forward(self, x1, other=1, other1=2):
        v1 = self.conv(x1)
        if other == False:
            other = torch.randn(v1.shape)
        v2 = v1 + other
        v3 = v2 + other1
        return v3
# Inputs to the model
x1 = torch.randn(1, 64, 64, 64)
