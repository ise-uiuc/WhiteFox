
class DepthWiseConvSigmoid(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.depthwiseConv = torch.nn.Conv2d(16, 16, (2,2), stride=2)
        # self.depthwiseConv = torch.nn.Conv2d(3, 8, 1, stride=2, padding=1)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.sig(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 16, 128, 128)
