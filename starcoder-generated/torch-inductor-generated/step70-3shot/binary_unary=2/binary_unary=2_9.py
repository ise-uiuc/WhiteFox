
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 5, stride=1, padding=0)
    def forward(self, x1):
        v1 = F.relu(x1)
        v2 = self.conv(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
