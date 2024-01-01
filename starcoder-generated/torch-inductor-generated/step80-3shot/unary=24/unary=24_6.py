
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(4, 2, (8, 2), stride=(2, 1), padding=1)
        self.conv1d = torch.nn.Conv1d(2, 4, (5, 1), stride=3, padding=2)
    def forward(self, x):
        negative_slope = 0.59877
        v1 = self.conv2d(x)
        v2 = v1 > 0
        v3 = v1 * negative_slope
        v4 = torch.where(v2, v1, v3)
        v5 = self.conv1d(v4)
        v6 = v5 > 0
        v7 = v5 * negative_slope
        v8 = torch.where(v6, v5, v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 4, 64, 64)
