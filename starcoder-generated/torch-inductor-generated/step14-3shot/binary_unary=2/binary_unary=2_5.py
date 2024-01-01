
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=2, padding=3)
        self.pading_2 = torch.nn.ZeroPad2d(padding=(3, 0, 2, 0))
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - v1
        v3 = self.pading_2(v2)
        v4 = F.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
