
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(4, 4, 3)
        self.norm = torch.nn.BatchNorm3d(4)
    def forward(self, x2):
        x = self.conv(x2)
        x = self.norm(x)
        x = self.conv(x)
        return x
# Inputs to the model
x2 = torch.randn(1, 4, 4, 3, 3)
