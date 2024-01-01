
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 4, stride=1, padding=0)
    def forward(self, x1):
        v1_1 = self.conv(x1)
        v1_2 = self.conv(x1)
        v1 = v1_1 + v1_2
        v2_1 = self.conv(x1)
        v2_2 = self.conv(x1)
        v2 = v2_1 + v2_2
        v3_1 = self.conv(x1)
        v3_2 = self.conv(x1)
        v3 = v3_1 + v3_2
        v4 = torch.relu(v1 + v2 + v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
