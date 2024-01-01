
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 5, stride=1, padding=2)
    def forward(self, x):
        v1 = self.conv1(x)
        return v1
# Inputs to the model
x = torch.randn(1, 16, 64, 64)
