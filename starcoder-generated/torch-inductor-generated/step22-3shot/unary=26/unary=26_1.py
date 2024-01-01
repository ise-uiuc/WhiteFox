
class Model(torch.nn.Module):
    def __init__(self"):
        super().__init__()
        self.conv1d = torch.nn.Conv1d(16, 7, 2,  stride=2)
        self.conv2d = torch.nn.Conv2d(33, 11, (1, 2), stride=1, padding=0, bias=False)
        self.conv3d = torch.nn.Conv3d(14, 21, (1, 2, 2), stride=1, padding=0, bias=False)
        self.conv4d = torch.nn.ConvTranspose3d(21, 14, (1, 2, 2), stride=1, padding=0, bias=False)
        self.conv5d = torch.nn.ConvTranspose3d(1, 3, (2, 2, 2), stride=2, padding=0)
        self.negative_slope = 0.0001
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x4):
        v1 = self.conv1d(x4)
        v2 = self.conv2d(v1)
        v3 = self.conv3d(x4)
        v4 = self.conv4d(v3)
        v5 = self.conv5d(v4)
        v6 = v5 > 0
        v7 = v5 * self.negative_slope
        v8 = torch.where(v6, v5, v7)
        v9 = self.sigmoid(v8)
        return v9
# Inputs to the model
x4 = torch.randn(4, 16, 20)
