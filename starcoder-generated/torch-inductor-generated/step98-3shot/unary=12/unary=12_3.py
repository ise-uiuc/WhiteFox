
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(64, 32, 1, stride=1, padding=0, bias=False)
        self.batch_norm = torch.nn.BatchNorm2d(32) # Change from False to True
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.batch_norm(v1)
        v3 = v1.sigmoid()
        v4 = v3 * v2
        return v4
# Inputs to the model
x1 = torch.randn(1, 64, 200, 300)
