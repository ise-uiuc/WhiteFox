
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convT1 = torch.nn.ConvTranspose2d(3, 64, 3, stride=5, padding=2, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64, momentum=0.8)
    def forward(self, x1):
        v1 = self.convT1(x1)
        v2 = self.bn1(v1)
        v3 = torch.sigmoid(v2)
        return v3
# Inputs to the model
x1 = torch.randn(2, 3, 64, 64)
