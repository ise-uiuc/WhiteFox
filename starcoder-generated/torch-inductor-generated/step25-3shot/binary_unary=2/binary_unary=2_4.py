
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=3, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - torch.rand(v1.shape)
        v3 = F.relu(v1)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
