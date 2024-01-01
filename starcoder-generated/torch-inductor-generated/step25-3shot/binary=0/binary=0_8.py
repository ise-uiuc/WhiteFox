
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(64, 3, 1, stride=1, padding=1)
    def forward(self, x1, bias1=True, other=True, size1=None):
        v1 = self.conv(x1)
        if bias1 == True or other == True or size1 == True:
            other = torch.randn(v1.shape)
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(32, 64, 4, 4)
