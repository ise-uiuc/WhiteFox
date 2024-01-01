
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 3, 3, stride=1, padding=1)
        self.relu = torch.nn.ReLU()
    def forward(self, x1, other=None):
        v1 = self.conv(x1)
        v2 = self.relu(v1)
        if other == None:
            other = torch.zeros(v2.shape)
        v3 = v2 + other
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
