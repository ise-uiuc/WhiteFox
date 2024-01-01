
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1[0]
        v3 = v2[0]
        v4 = v2[1]
        v5 = v2[2]
        v6 = v2[3]
        v7 = v3 + v4 + v5 + v6
        v8 = torch.relu(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
