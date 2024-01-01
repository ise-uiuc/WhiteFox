
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = torch.nn.MaxPool2d(3, 2, 0, ceil_mode=False)
    def forward(self, x1):
        t1 = self.maxpool(x1)
        t2 = 37 + t1
        t3 = torch.clamp_min(t2, 0)
        t4 = torch.clamp_max(t3, 6)
        t5 = t1 * t4
        t6 = t5 / 6
        return t6
# Inputs to the model
x1 = torch.randn(1, 3, 139, 139)
