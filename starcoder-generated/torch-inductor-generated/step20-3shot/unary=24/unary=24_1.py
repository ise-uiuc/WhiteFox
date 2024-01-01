
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.negative_slope = negative_slope

    def forward(self, x):
        # Conv1:
        v1 = self.conv1(x)
        v2 = v1 > 2
        v3 = v1 * self.negative_slope
        v4 = torch.where(v2, v1, v3)
        # Conv2:
        v5 = self.conv2(x)
        v6 = v5 > 2
        v7 = v5 * self.negative_slope
        v8 = torch.where(v6, v5, v7)
        # Concat
        v9 = torch.cat((v4, v8), 1)
        v10 = self.conv1(v9)
        return v10
# Inputs to the model
negative_slope = 0.2
x1 = torch.randn(1, 3, 64, 64)
