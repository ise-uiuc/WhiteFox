
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 6, 2, stride=2, padding=1)
    def forward(self, x1, padding2=None, stride2=None):
        v1 = self.conv(x1)
        if stride2 == None:
            stride2 = torch.randn(v1.shape)
        v2 = v1 + padding2
        return v2
# Inputs to the model
x1 = torch.randn(1, 4, 32, 32)
