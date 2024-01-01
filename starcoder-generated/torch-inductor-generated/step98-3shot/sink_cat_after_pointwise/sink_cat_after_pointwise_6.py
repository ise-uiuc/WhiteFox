
torch.manual_seed(0)
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(5, 5, 1)
        self.conv2 = torch.nn.Conv2d(5, 5, 1)
        self.pool = torch.nn.MaxPool2d(2, 2)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        y = torch.cat([x, x], dim=0)
        y = y.relu()
        z = torch.cat([x, y], dim=0)
        z = z.relu()
        return z
# Inputs to the model
x = torch.randn(2, 5, 4, 4)
