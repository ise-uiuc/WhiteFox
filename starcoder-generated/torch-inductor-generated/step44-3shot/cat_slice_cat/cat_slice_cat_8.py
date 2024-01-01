
class Module1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.v1 = torch.nn.Linear(1024, 2)

    def forward(self, t):
        v1 = self.v1(t)
        return v1

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m1 = Module1()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
 
    def forward(self, x1, x2):
        v1 = torch.nn.functional.adaptive_avg_pool2d(x2, [1, 1])
        v1 = v1.reshape(v1.size(0), -1)
        v2 = self.m1(v1)
        v3 = torch.cat([v2.unsqueeze(1), x1], dim=1)
        v4 = v3[:, 0, :]
        v5 = self.conv(v4.unsqueeze(1))
        v6 = v3 * 0.9793241061191791
        v7 = v5 * v6
        return v7

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 128, 128)
