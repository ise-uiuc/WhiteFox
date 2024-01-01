
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 1)
 
    def forward(self, x1):
        v2 = x1.permute(0, 2, 3, 1)
        v1 = self.linear(v2)
        v3 = v1 * 0.5
        v4 = v1 + (v1 * v1 * v1) * 0.044715
        v5 = v4 * 0.7978845608028654
        v6 = torch.tanh(v5)
        v7 = v6 + 1
        v8 = v3 * v7
        return v8


# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4, 8, 8)
