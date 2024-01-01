
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(5, 62, 5, padding=10, bias=False)
        self.bn = torch.nn.BatchNorm2d(64, affine=True)
        self.relu = torch.nn.ReLU()
    def forward(self, x9):
        v4 = self.conv_t(x9)
        v2 = self.bn(v4)
        v3 = v2 > 0
        v1 = v2 * 0.01
        v5 = torch.where(v3, v2, v1)
        return self.relu(v5)
# Inputs to the model
x9 = torch.randn(4, 5, 86, 61)
