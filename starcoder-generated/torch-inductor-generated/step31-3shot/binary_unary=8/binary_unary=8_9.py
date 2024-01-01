
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=2, padding=0)
        self.conv_batchn = torch.nn.BatchNorm2d(8, affine=False, momentum=None, eps=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv_batchn(v1)
        v3 = torch.relu(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
