
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__(self)
        self.conv = torch.nn.Conv2d(512, 512, 1, 1, 0, bias=False)
        self.BN_ReLU = torch.nn.BatchNorm2d(512)
        self.relu = torch.nn.ReLU(inplace=True)
        self.convD1 = torch.nn.ConvTranspose2d(512, 512, 1, 2, 0, 1, bias=False)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.BN_ReLU(v1)
        v3 = self.relu(v2)
        v4 = self._forward_impl(v3)

        return v4

    def _forward_impl(self, v1):
        v2 = self.convD1(v1)
        v3 = self.sigmoid(v2)

        return v3
# Inputs to the model
x1 = torch.randn(1, 512, 4, 4)
