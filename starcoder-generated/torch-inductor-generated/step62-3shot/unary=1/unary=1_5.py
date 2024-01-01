
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1024, 3)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 * 0.5
        v1 = v1.repeat(2, 0)
        v3 = v1 * v1 * v1
        v3 = v3 * 0.044715
        v4 = v3 + v2
        v5 = v4 * 0.7978845608028654
        v6 = torch.tanh(v5)
        v7 = v6 + 1
        v8 = v1 * v7
        return v8


# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 1024)
