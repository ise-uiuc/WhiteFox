
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l = torch.nn.Linear(512, 512)
 
    def forward(self, x2):
        y = self.l(x2)
        v1 = y * 0.5
        v2 = y + y * (y * y) * 0.044715
        v3 = v2 * 0.7978845608028654
        v4 = torch.tanh(v3)
        v5 = v4 + 1
        v6 = v1 * v5
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 512)
