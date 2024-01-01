
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(4, 8, (5,5), stride=2, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - 6
        v3 = torch.abs(v2)
        v4 = torch.nn.functional.prelu(v3, 0.1)
        return v4
# Inputs to the model
x1 = torch.randn(3, 4, 16, 16)
