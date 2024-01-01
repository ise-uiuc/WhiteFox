
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 7, 1, stride=1, padding=1)
    def forward(self, x1, other=1, padding1=None):
        v1 = self.conv(x1)
        if None in (padding1, padding2):
            v2 = (v1*v1)/v1
            return torch.transpose(v1, 1, 1)
        else:
            v2 = (v1-v1)*1
            return v2
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
