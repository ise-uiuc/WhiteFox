
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.nn.ReLU(v1)
        v2 = v1 + v2
        v3 = torch.nn.ReLU(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 80, 80)
