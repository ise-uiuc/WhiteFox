
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 7)
    def forward(self, b1, b2=None, padding0=None):
        v1 = self.conv(b1)
        if b2 == None:
            b2 = torch.randn(v1.shape)
        v2 = v1 + b2
        return v2
# Inputs to the model
b1 = torch.randn(1, 1, 7, 7)
