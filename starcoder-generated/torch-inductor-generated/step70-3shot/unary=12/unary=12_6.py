
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3, stride=3, padding=1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = F.relu(v1)
        v3 = F.sigmoid(v1)
        v4 = torch.relu(v3)
        v5 = v1 * v2
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
