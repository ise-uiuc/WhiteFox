
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(6, 8)
 
    def forward(self, x1):
        w1 = self.linear(x1)
        w2 = w1 * 0.5
        w3 = w1 * 0.7071067811865476
        w4 = torch.erf(w3)
        w5 = w4 + 1
        w6 = w2 * w5
        return w6

# Initializing model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 6)
