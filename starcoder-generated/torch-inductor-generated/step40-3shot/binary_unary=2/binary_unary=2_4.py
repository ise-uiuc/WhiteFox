
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - 1.0
        v3 = F.relu(v2)
        v4 = torch.nn.AvgPool2d(kernel_size=4, stride=4, padding=0)(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
