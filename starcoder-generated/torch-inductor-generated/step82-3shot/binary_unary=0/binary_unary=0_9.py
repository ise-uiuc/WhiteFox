
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3,3,3, padding=1, bias=False)
        self.norm1 = torch.nn.InstanceNorm3d(3)
        self.conv2 = torch.nn.Conv2d(3,3,3, padding=1, bias=False)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.norm1(v1)
        v3 = self.conv2(v2)
        return v3
# Inputs to the model
x = torch.randn(2, 3, 32, 32)
