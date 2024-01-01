
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(3, 8, 1, stride=1, padding=1)
        self.other_conv = torch.nn.Conv1d(8, 8, 9)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = 3 + v1
        v3 = v2.relu6()
        v4 = v3.normalize(1)
        v5 = self.other_conv(v4)
        v6 = 3 + v5
        v7 = v6.relu6()
        v8 = v7.normalize(1)
        return v8
# Inputs to the model
x1 = torch.randn(5, 3, 64, 64, 64)
