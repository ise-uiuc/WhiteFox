
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 4, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.relu(v1)
        v3 = torch.maxpool2d(v2, kernel_size=5)
        v4 = torch.relu(v3)
        v5 = v4[:-1,:-1]
        v6 = v4[1:,:-1]
        v7 = torch.relu(torch.add(v5, v6))
        del v5, v6
        return v7
# Inputs to the model
x1 = torch.randn(1, 4, 128, 128)
