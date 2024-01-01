
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(11, 5)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 * 0.5
        v3 = v1 * v1 * v1 + 0.044715 * v1
        v4 = 0.7978845608028654 * v3
        v5 = torch.tanh(v4)
        return v1 * (v5 + 1.0)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 11)
