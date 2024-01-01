
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 12, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(12, 24, 3, padding=1)
    def forward(self, x1):
        v1 = F.pad(x1, (4, 4, 1, 1), "constant", 1)
        v2 = x1.transpose(1, 3)
        v3 = self.conv1(v1)
        v4 = self.conv2(v3)
        v5 = torch.flatten(v4)
        return v5
# Inputs to the model
x1 = torch.randn(3, 1, 4, 4)
