
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 1, 1, stride=1, padding=1, bias=False)
    def forward(self, x1):
        x2 = self.conv1(x1)
        x3 = x2 - 0.5
        x4 = F.relu(x3)
        return x4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
