
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.conv = torch.nn.Conv2d(3, 3, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.relu(x1)
        v2 = self.conv(v1)
        v3 = F.leaky_relu(v2, negative_slope=0.2, inplace=True)
        v4 = torch.sigmoid(v3)
        v5 = torch.sum(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
