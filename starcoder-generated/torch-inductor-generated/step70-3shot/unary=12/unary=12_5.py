
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(5, 3, 3, stride=1, padding=1)
        self.relu = torch.nn.ReLU(inplace=False)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.relu(v1)
        # v3 = F.sigmoid(v2)
        # v4 = v2 * v3
        return v2
# Inputs to the model
x1 = torch.randn(1, 5, 64, 64)
