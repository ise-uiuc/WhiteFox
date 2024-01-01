
class NonsenseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(input, output, 1, stride=1, padding=0)
    def forward(self, x1=None, other=None):
        v1 = self.conv1(x1)
        if other == None:
            other = torch.randn(v1.shape)
        v2 = v1 + other
        v3 = v2 + other
        v4 = v3 + other
        v5 = v4 + other
        v6 = v5 + other
        return v6
# Inputs to the model
x1 = torch.randn(1, 2, 64, 64)
