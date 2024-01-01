
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(6, 18, 1, stride=2, padding=1)
    def forward(self, x1, padding1=None):
        v1 = self.conv(x1)
        if padding1 == None:
            padding1 = torch.randn(v1.shape)
        v2 = torch.nn.functional.relu6(v1 + padding1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 6, 64, 64)
