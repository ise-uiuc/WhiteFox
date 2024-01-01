
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, 1, 1)
        self.relu = torch.nn.ReLU(None)
    def forward(self, x1):
        t1 = self.conv1(x1)
        t2 = self.relu(t1)   # Non-functional ReLU.
        t3 = t2 * t1           # Non-functional ReLU.
        t4 = self.relu(t3)
        return t4
# Inputs to the model
x1 = torch.randn(2, 3, 64, 64)
