
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(952, 336)
 
    def forward(self, x1):
        v1 = x1.flatten(1, -1)
        v2 = self.linear(v1)
        v3 = torch.clamp(v2, max=6) + 3
        v4 = v3 * 0.16666666666666666
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 96, 88)
