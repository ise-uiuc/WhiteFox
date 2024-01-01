
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(9216, 2)
 
    def forward(self, X1):
        v1 = self.linear(X1)
        v2 = v1*0.5
        v3 = v1*0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        return v6

# Initializing the model
m = Model()

# Inputs to the model
__inputs__ = torch.rand(1, 9216)
