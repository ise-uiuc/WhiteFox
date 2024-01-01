
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        u2 = v1 * 0.5
        u3 = v1 + (v1 * v1 * v1) * 0.044715
        u4 = u3 * 0.7978845608
        u5 = torch.tanh(u4)
        u6 = u5 + 1
        u7 = u2 * u6
        return u7

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
