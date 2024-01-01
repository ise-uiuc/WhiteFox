
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 16, bias=True) * 0.1
 
    def forward(self, x1):
        v1 = x1
        v2 = self.linear(v1)
        v2 = torch.mul(v2, 0.5)
        v3 = torch.pow(v1, 3)
        v3 = torch.mul(v3, 0.044715)
        v3 = torch.add(v2, v3)
        v4 = torch.mul(v3, 0.7978845608028654)
        v5 = torch.tanh(v4)
        v5 = torch.add(v5, 1)
        v5 = torch.mul(v2, v5)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(4, 10)
