
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 2, stride=2, padding=4)
    def forward(self, x1):
        v1 = self.conv(x1)
        r1 = torch.nn.functional.elu(v1)
        v2 = 2
        v3 = v1 + v2
        v4 = v1 / v3
        v5 = torch.nn.functional.relu(v4)
        v6 = r1 + v5
        return v6
# Inputs to the model
x1 = torch.randn(2, 3, 28, 28)
