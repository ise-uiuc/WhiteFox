
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(15, 25, 3, stride=2, padding=1)
    def forward(self, x1, other=3, other1=4):
        v1 = self.conv(x1)
        if other == False:
            other = torch.randn(v1.shape)
        v2 = v1 + other
        v3 = v2 + other1
        return v2
# Inputs to the model
x1 = torch.randn(1, 15, 128, 128)
