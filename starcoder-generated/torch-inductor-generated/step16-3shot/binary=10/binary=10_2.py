
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 64)
 
    def forward(self, x1, others):
        v0 = x1
        v1 = self.linear(v0)
        v2 = v1 + others
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 32)
others = torch.randn(1, 16)
