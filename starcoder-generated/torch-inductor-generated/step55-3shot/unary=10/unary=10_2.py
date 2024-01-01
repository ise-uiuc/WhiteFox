
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(20, 25)
 
    def forward(self, x2):
        v7 = self.linear(x2)
        v8 = v7 + 3.0
        v9 = torch.clamp_min(v8, 0.0)
        v10 = torch.clamp_max(v9, 6.0)
        v11 = v10 / 6.0
        return v11

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(2, 20)
