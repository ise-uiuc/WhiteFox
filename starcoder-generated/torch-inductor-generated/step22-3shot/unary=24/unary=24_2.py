
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(4, 4, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(4, 4, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(4, 4, 1, stride=1, padding=0)
        self.batch_normalization = torch.nn.BatchNorm1d(4, 0.9936906674194336)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        negative_slope = 0.12678904724121094
        v3 = self.conv3(v2)
        v4 = v3 > 0
        v5 = v3 * negative_slope
        v6 = torch.where(v4, v1, v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 4, 32, 32)
