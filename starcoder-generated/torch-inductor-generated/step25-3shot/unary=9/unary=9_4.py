
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.relu = torch.nn.ReLU(inplace=True)
    def forward(self, x1):
        t1 = self.conv1(x1)
        t2 = self.relu(t1)
        t3 = self.relu(t2)
        t4 = self.relu(t3)
        return t4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
