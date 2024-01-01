
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(32, 11, 13, padding=2, stride=1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x = torch.randn(1, 32, 45, 129)
