
# Please change the name of this class
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
    def forward(self, x1, padding1=1):
        v1 = self.conv(x1)
        if padding1 == 1:
            padding1 = torch.randn(v1.shape[:-1])
        else:
            padding1 = torch.randn(v1.shape[:-1])
        v2 = v1 + padding1
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
