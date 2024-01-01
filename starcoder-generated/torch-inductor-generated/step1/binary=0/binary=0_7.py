
class M1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(10, 16, 3, stride=1, padding=1)
 
    def forward(self, x, o):
        v11 = self.conv(x)
        v12 = v11 + o
        return v12

# Initializing the model
m1 = M1()

# Inputs to the model
x = torch.randn(1, 10, 64, 64)
o = torch.ones(1, 16, 64, 64)
