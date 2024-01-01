
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU(inplace=False)
        self.conv = torch.nn.Conv2d(3, 2, 1, stride=1, padding=1)
    def forward(self, x1, other=None):
        v1 = self.relu(x1)
        v2 = self.conv(v1)
        if other == None:
            other = torch.randn(v2.shape)
        v3 = v2 + other
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
