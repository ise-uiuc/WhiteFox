
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)
 
    def forward(self, o1):
        v0 = torch.neg(o1)
        v2 = v0 - self.linear.weight
        return v2

# Initializing the model
m = Model()

# Inputs to the model
o1 = torch.randn(1, 8)
