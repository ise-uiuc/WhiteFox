
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.upsample_bilinear2d = torch.nn.Upsample(mode='bilinear', scale_factor=2.0)
        self.conv2d_1 = torch.nn.Conv2d(3, 64, (1, 3), stride=(1, 1), padding=(0, 1))
        self.conv2d_2 = torch.nn.Conv2d(64, 1, (1, 2), stride=(1, 1), padding=(0, 0))
        self.upsample_bilinear2d_1 = torch.nn.Upsample(mode='bilinear', scale_factor=2.0)
    def forward(self, x1):
        v1 = self.upsample_bilinear2d(x1)
        v2 = self.conv2d_1(v1)
        v3 = torch.relu(v2)
        v4 = self.conv2d_2(v3)
        v5 = torch.sigmoid(v4)
        v6 = self.upsample_bilinear2d_1(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 1, 16)
