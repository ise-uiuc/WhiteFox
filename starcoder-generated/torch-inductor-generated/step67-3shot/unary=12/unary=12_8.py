
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(5, 2, 1, stride=1, padding=0, bias=False)
        self.sigmoid = torch.nn.Sigmoid()
        self.linear = torch.nn.Linear(2, 8)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.sigmoid(v1)
        v3 = v1 * v2
        v3 = self.linear(v3)
        return v3
# Inputs to the model
x1 = torch.randn(1, 5, 64, 64)
