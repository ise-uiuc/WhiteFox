
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.conv = torch.nn.Conv3d(2, 5, 5)
        torch.manual_seed(1)
        self.bn = torch.nn.BatchNorm3d(5)
    def forward(self, x2):
        y2 = self.conv(x2)
        y2 = self.bn(y2)
        return y2 + y2
# Inputs to the model
x2 = torch.randn(1, 2, 2, 2, 2)
