
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 5, stride=2, padding=1)
    def forward(self, x2):
        v1 = self.conv(x2)
        v1 = v1 - 3
        v1 = F.relu(v1)
        v2 = v1.permute(0, 1, 3, 2)
        return v2
# Inputs to the model
x2 = torch.randn(1, 3, 16, 32)
