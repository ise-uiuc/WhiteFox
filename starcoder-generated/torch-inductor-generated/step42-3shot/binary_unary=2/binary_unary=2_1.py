
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(10, 1, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - 10
        return F.relu(v2)
# Inputs to the model
x1 = torch.randn(1, 10, 128, 128)
