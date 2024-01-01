
class Module0(torch.nn.Module):
    def __init__(self, ):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(8, 3, 3, stride=1, padding=1)
 
    def forward(self, x):
        v2 = self.conv_t(x)
        v3 = v2 * 0.5
        v4 = v2 * 0.7071067811865476
        v5 = torch.erf(v4)
        v6 = v5 + 1
        v7 = v3 * v6
        return v7

class Module1(torch.nn.Module):
    def __init__(self, ):
        super().__init__()
        self.module0 = Module0()
 
    def forward(self, x):
        v2 = self.module0(x)
        v3 = v2 * 0.5
        v4 = v2 * 0.7071067811865476
        v5 = torch.erf(v4)
        v6 = v5 + 1
        v7 = v3 * v6
        return v7

# Initializing the model
m = Module1()

# Inputs to the model
x = torch.randn(1, 8, 64, 64)
