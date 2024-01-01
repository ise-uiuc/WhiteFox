
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d((31 + 1), 16, 3)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = (0.00000__000000__000000__000000__).float()
        v3 = v1 + v2
        v4 = torch.cat([v1, v3, v1, v3], 1)
        v5 = v4.relu()
        v6 = v5.permute(0, 1, 3, 2)
        return v6
# Inputs to the model
x1 = torch.randn(1, 31, 24, 24)
