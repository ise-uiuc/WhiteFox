
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 8)
 
    def forward(self, x2):
        v7 = self.linear(x2)
        v8 = v7 * torch.clamp(v7 + 3, 0, 6)
        v9 = v8 / 6
        return v9

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 64, 64)
