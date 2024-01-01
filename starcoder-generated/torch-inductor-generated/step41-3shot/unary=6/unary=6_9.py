
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3, stride=1, padding=1)
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = t1 + 3
        t3 = t2.clamp(0, 6)
        t4 = t1 * t3
        t5 = t4 / 6
        t6 = self.relu(t5)
        return t6
# Inputs to the model
x1 = torch.randn(2, 3, 64, 64)
