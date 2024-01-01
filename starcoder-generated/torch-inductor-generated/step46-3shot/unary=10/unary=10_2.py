
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 6)
 
    def forward(self, x1):
        w1 = self.linear(x1)
        w2 = w1 + 3
        w3 = torch.clamp(w2, 0, 6)
        w4 = w3 / 6
        return w4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
