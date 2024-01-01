
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(256, 256, 3, stride=1, padding=1)
    def forward(self, u, key):
        t1 = self.conv1(key)
        t2 = self.conv1(key)
        t3 = self.conv1(key)
        t4 = u + t1
        t5 = u + t2
        t6 = u + t3
        t7 = t4 + t5
        t8 = t7 + t6
        return t8
# Inputs to the model
u = torch.randn(1, 256, 128, 128)
key = torch.randn(1, 256, 64, 64)
