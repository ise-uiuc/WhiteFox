
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.sigmoid = torch.nn.Sigmoid()
        self.conv1 = torch.nn.Conv2d(8, 12, 1, stride=1, padding=0, bias=True)
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.sigmoid(v1)
        v3 = v1 * v2

        v4 = self.conv1(v3)
        v5 = self.relu(v4)
        v6 = v5 * v4
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 110, 110)
