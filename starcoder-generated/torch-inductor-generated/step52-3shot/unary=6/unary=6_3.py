
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 32, 1, stride=2, padding=0)
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = t1 + 3
        t3 = t2.clamp(0, 6)
        t4 = t1 * t3
        t5 = t4 / 6
        t6 = self.relu(conv)
        return t6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
