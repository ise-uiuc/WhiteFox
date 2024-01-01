
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x1):
        w1 = self.linear(x1)
        w2 = torch.clamp(w1 + 3, min=0, max=6)
        w3 = w2 / 6
        return w3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
