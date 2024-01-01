
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(100, 10)
 
    def forward(self, x1):
        w1 = self.linear(x1)
        w2 = w1 + 3
        w3 = torch.clamp_min(w2, 0)
        w4 = torch.clamp_max(w3, 6)
        w5 = w4 / 6
        return w5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 100)
