
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        if False:
            self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        else:
            self.linear = torch.nn.Linear(8 * 64 * 64, 1)
 
    def forward(self, x1):
        if False:
            v1 = self.conv(x1)
        else:
            v1 = self.linear(torch.flatten(x1, start_dim=1))
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
