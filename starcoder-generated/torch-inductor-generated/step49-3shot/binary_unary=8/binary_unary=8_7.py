
class Model(torch.nn.Module):
    # Note that padding = 3 would be an example of a valid padding argument
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(2, 8, 3, stride=2, padding=3)
        self.conv2 = torch.nn.Conv2d(2, 8, 3, stride=2, padding=3)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = self.conv2(x1)
        v4 = self.conv1(x1)
        v5 = v1 + v2 + v3 + v4
        v6 = torch.relu(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 2, 64, 64)
