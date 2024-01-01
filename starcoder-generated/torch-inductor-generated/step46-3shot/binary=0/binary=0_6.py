
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1, other=2, padding1=1, padding2=1, padding3=1, padding4=1, padding5=1, padding6=1, padding7=1):
        v1 = self.conv(x1)
        v2 = v1 + 2
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
