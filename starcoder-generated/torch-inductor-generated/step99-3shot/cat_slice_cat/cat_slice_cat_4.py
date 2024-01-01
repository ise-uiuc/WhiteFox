
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
 
    def forward(self, x1):
        v1 = self.___concat___1([x1, x1, x1])
        __slice1___1__ = v1[:, 0:9223372036854775807]
        __slice1___2__ = __slice1___1__[:, 0:size] 
        v2 = self.conv(v1)
        v3 = v2 * 0.5
        v4 = __slice1___2__ * 0.7071067811865476
        v5 = v4 + 1
        v6 = v3 * v5
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
