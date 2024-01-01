
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers0 = torch.nn.Linear(256, 128)
        self.layers1 = torch.nn.Linear(128, 256)
 
    def forward(self, x1):
        v1 = self.layers0(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = self.layers1(v6)
        return v7

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 256)
