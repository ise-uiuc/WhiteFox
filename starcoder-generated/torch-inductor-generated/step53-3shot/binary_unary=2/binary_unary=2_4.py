
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 5, stride=5, padding=2)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - 512
        v3 = F.relu(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
