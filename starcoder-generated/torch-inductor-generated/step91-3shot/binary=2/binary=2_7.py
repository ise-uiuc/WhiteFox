
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.linear = nn.Linear(1, 1, bias=False)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - v1
        v3 = v2 - v1
        # v3 has the same shape and content as v1
        v4 = v3 - 0.0
        # v4 has the same shape and content as v1
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
