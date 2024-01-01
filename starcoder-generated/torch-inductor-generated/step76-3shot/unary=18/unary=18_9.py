
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, kernel_size=3)
        self.conv2 = torch.nn.Conv2d(8, 8, kernel_size=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = torch.sigmoid(v1)
        v4 = torch.sigmoid(v2)
        v5 = torch.cat([v3, v4], 0)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 28, 28)
