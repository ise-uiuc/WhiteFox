
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.ln1 = torch.nn.LayerNorm((16, 16, 16), elementwise_affine=True)
        self.bn1 = torch.nn.BatchNorm3d(16)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = self.conv3(x2)
        v4 = self.ln1(v1 + v2 + v3)
        v5 = self.bn1(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 16, 16, 16)
x2 = torch.randn(1, 3, 16, 16, 16)
