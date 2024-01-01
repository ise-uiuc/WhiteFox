
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.conv = torch.nn.Conv3d(2, 2, 2)
        torch.manual_seed(1)
        self.bn = torch.nn.BatchNorm3d(2)
    def forward(self, x3):
        s3 = self.bn(x3)
        s3 = self.conv(s3)
        s3 = self.bn(s3)
        s3 = self.conv(s3)
        s3 = self.bn(s3)
        return s3
# Inputs to the model
x3 = torch.randn(1, 2, 2, 100, 10)
