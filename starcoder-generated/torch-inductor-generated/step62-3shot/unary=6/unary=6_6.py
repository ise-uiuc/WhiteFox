
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 2, stride=1, padding=1, dilation=1)
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = t1 + torch.full_like(t1, 3, dtype=torch.float)
        t3 = torch.nn.functional.relu6(t2)
        t4 = torch.nn.functional.relu(t3)
        t5 = t1 * t4
        t6 = t5 / 6
        return t6
# Inputs to the model
x1 = torch.randn(2, 3, 64, 64)
