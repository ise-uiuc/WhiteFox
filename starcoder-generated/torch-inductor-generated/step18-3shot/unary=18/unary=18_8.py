
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = torch.nn.functional.max_pool2d(x1, kernel_size=[2, 2], stride=2, padding=0)
        v2 = self.conv(v1)
        v3 = torch.sigmoid(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
