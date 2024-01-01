
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 7, 5, stride=1, padding=2)
        self.relu = torch.nn.ReLU(inplace=False)
    def forward(self, x):
        negative_slope = 0.10669922
        v1 = self.conv(x)
        v2 = self.relu(v1)
        v3 = v2 > 0
        v4 = v2 * negative_slope
        v5 = torch.where(v3, v2, v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 4, 90, 10)
