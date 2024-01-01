
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1 + 1, 8, 1 * 1 * 1, stride=1)
        )
    def forward(self, x1, padding1=None):
        if padding1 == None:
            padding1 = torch.randn(x1.shape)
        v1 = self.conv(x1 + padding1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 2, 64, 64)
