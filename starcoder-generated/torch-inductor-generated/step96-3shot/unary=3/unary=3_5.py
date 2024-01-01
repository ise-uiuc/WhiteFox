
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 11, padding=5, stride=1, bias=False)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = self.conv(x1)
        return self.sigmoid(v1)
# Inputs to the model
x1 = torch.randn(1, 3, 73, 80)
