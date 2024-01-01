
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(6, 6)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        def clamped(x2):
            if x2 > 6:
                return torch.tensor(6, dtype=torch.float32)
            elif x2 < 0:
                return torch.tensor(0, dtype=torch.float32)
            else:
                return x2
        v2 = v1 * clamped(v1+3)
        v3 = v2 / 6
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 6)
