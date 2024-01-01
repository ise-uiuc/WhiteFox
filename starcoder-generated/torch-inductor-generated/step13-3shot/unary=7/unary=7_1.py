
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.min(v1 + 3, torch.tensor(6.0))
        v3 = v1 * v2
        return v3 / 6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
