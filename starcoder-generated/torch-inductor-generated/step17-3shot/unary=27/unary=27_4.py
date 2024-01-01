
class Model(torch.nn.Module):
    def __init__(self, min_value=1.41421):
        super().__init__()
        self.relu6 = torch.nn.ReLU6(min_value=min_value)
        self.conv = torch.nn.Conv2d(5, 7, 2, stride=1)
    def forward(self, x1):
        v1 = self.relu6(x1)
        v2 = self.conv(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 5, 152, 152)
