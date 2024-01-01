
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 2, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 16, 1, stride=1, padding=1)
    def forward(self, input1):
        v1 = 1 + self.conv1(input1)
        v2 = v1 - 1
        v3 = v2.clamp_min(0)
        v4 = v3.clamp_max(2)
        v5 = v4.div(2)
        v6 = self.conv2(input1)
        v7 = v5 + v6
        v8 = v7 + 3
        v9 = v8.clamp_min(0)
        v10 = v9.clamp_max(6)
        v11 = v10.div(6)
        return v11
# Inputs to the model
input1 = torch.randn(1, 3, 64, 64)
