
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.conv = torch.nn.Conv3d(3, 3, 1)
        torch.manual_seed(1)
        self.bn = torch.nn.BatchNorm3d(3)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
# Inputs to the model
x = torch.randn(1, 3, 3, 3, 3)
