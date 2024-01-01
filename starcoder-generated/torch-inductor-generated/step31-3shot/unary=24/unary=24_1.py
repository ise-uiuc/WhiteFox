
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(4, 8, 1, stride=1, padding=1)
        self.conv_2 = torch.nn.Conv2d(8, 4, 1, stride=1, padding=1)
    def forward(self, x):
        negative_slope = 0
        v1 = self.conv_1(x)
        v2 = torch.sum(v1, dim=1)
        v3 = v2 > 0
        v4 = v2 * negative_slope
        v5 = torch.where(v3, v2, v4)
        v6 = v5.shape[2]
        v7 = v5.shape[3]
        v8 = self.conv_2(v5)
        v9 = v8[:,:,:v6,:v7].squeeze()
        return v9
# Inputs to the model
x1 = torch.randn(1, 4, 64, 64)
