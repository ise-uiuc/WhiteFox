
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(4, 8, 1)
        self.conv_2 = torch.nn.Conv2d(8, 32, 1)
    def forward(self, x1):
        v1 = self.conv_1(x1)
        v2 = self.conv_2(v1)
        v3 = v2 - 8
        v4 = F.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 4, 96, 96)
