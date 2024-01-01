
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 10)
 
    def forward(self, x1, x2):
        b = x2.size(0)
        v1 = x1.reshape(b, 5, 1)
        v4 = x1.reshape(1, b, 5).transpose(-1, -2)
        v2 = self.linear(v1)
        v3 = self.linear(v4)
        v5 = v2 * 0.5
        v6 = v2 + (v2 * v2 * v2) * 0.044715
        v7 = v6 * 0.7978845608028654
        v8 = torch.tanh(v7)
        v9 = v8 + 1
        v10 = v5 * v9
        return v10

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 5)
x2 = torch.randn(2, 5)
