
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(64, 9, 3, stride=1, padding=1)
    def forward(self, x1, other1=1, other2=2, other3=3, padding1=None):
        v1 = self.conv(x1)
        if padding1 is None:
            padding1 = torch.randn(v1.shape)
        v2 = v1 + other1 + other2 + other3
        return v2
# Inputs to the model
x1 = torch.randn(1, 64, 32, 32)
