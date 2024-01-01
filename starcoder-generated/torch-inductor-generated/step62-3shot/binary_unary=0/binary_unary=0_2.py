
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 3, padding=1, groups=4)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = torch.transpose(v1, 2, 3)
        v3 = torch.relu(v2)
        v4 = torch.max(v3, 1)
        return v4[0], v4[1]
# Inputs to the model
x = torch.randn(1, 16, 64, 64)
