
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = torch.nn.AvgPool2d(5)
        self.conv = torch.nn.Conv2d(2, 4, 1, stride=1, padding=0, bias=False)
    def forward(self, x1):
        v1 = self.pool(x1)
        v2 = self.conv(v1)
        v3 = torch.nn.functional.relu(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 32, 32)
