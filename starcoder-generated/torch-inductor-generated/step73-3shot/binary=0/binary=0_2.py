
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(5, 13, 2, stride=2, padding=1, dilation=1)
    def forward(self, x1, other=None, other2=None):
        v1 = self.conv1(x1)
        if other2 == None:
            other2 = torch.randn(v1.shape)
        v2 = v1 + other2
        if other == None:
            other = torch.randn(v1.shape)
        v3 = v1 - other
        return v3
# Inputs to the model
x1 = torch.randn(1, 5, 32, 32)
