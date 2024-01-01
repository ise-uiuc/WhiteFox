
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 + 3
        print(v2.size())
        v3 = torch.nn.functional.relu6(v2)
        print(v3.size())
        v4 = v3.div(6)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
