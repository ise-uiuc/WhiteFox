
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x1):
        v1 = torch.mm(x1, m2.weight)
        v2 = v1 * 0.5
        v3 = v1 + v1 * v1 * v1 * 0.044715
        v4 = v3 * 0.7978845608028654
        v5 = torch.tanh(v4)
        v6 = v5 + 1
        v7 = v2 * v6
        return v7

# Initializing the model
m = Model()

# Initializing all weights
n = [0.25, 0.5, 0.75]
m2 = torch.nn.Linear(1, len(n), bias=False)
m2.weight[0][0] = n[0]
m2.weight[0][1] = n[1]
m2.weight[0][2] = n[2]

# Inputs to the model
x1 = torch.randn(1, 1)
