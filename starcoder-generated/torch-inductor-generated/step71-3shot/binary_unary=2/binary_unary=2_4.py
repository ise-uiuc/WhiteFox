
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(300, 7, 51, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - 12.5
        v3 = F.relu(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 300, 7, 1)
