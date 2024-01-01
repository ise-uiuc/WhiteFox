
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(73, 95, 1, stride=1, padding=1)
    def forward(self, x1, other=None, padding1=None, padding2=True, padding3='test', padding4=None, padding5=None, padding6=None, padding7=None):
        v1 = self.conv(x1)
        if other == None:
            other = torch.randn(95, v1.shape[2], v1.shape[3])
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(1, 73, 64, 64)
