
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(64, 128, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - 127
        v3 = F.relu(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 64, 56, 56)
