
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 5)
        self.pool = torch.max()
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = F.relu6(x)
        return x - 42
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
