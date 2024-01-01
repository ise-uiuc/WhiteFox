
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 8, 8, stride=8, padding=8)
    def forward(self, x1, other=False):
        v1 = self.conv(x1)
        if other == False:
            other = torch.randn(v1.shape)
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(1, 8, 2048, 2048)
