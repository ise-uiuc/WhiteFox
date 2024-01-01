
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 16, 1, stride=1, padding=1)
    def forward(self, x1, padding1=True, padding2=66, padding3=0.575723, padding4=-2.893867, padding5=[-0.1244, 2.837, -3.7528, 6.8744]):
        v1 = self.conv(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 1, 14, 112)
