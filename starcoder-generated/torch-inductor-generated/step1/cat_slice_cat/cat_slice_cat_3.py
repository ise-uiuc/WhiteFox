
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
 
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 + 1
        v3 = v1 * 0.5
        v4 = v2 * 0.7071067811865476
        v5 = torch.erf(v4)
        v6 = v5 + 1
        v7 = torch.cat((v3, v6), 1)
        v8 = v7[1:9223372036854775807:1]
        return v8

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
