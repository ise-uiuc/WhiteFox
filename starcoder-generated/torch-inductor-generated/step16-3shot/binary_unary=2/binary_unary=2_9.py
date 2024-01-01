
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(3, 20, 5, stride=2, padding=2, bias=False)
        self.conv_2 = torch.nn.Conv2d(3, 20, 3, stride=1, padding=1, bias=False)
    def forward(self, x1):
        v1 = self.conv_1(x1)
        v2 = self.conv_2(x1)
        v3 = v1 + v2
        v4 = F.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
