
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 1)
    def forward(self, x1):
        t1 = self.conv1(x1)
        return torch.relu(t1 + 3)
# Inputs to the model
x1 = torch.randn(2, 3, 28, 28)
