
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 32, 17, stride=1, padding=8)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - v1
        v3 = v1 * 0.7
        v4 = torch.div(v1, v3)
        v5 = F.relu(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 80, 80)
