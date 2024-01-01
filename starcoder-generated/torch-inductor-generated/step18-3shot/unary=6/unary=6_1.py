
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 3, 1, stride=1, padding=1)
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = 3 - t1
        t3 = t2.clamp(-3, 3)
        t4 = t3 * t1
        t5 = t4 * 3 
        return t5
# Inputs to the model
x1 = torch.randn(2, 4, 64, 64)
