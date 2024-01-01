
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(32, 64, 3, stride=1, bias=False)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - 50
        v3 = F.gelu(v2)
        v4 = v3.transpose(0, 1)
        return v4
# Inputs to the model
x1 = torch.randn(4, 32, 28, 28)
