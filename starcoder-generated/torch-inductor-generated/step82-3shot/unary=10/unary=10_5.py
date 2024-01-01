
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)
 
    def forward(self, x1):
        h1 = self.linear(x1)
        h2 = h1 + 3
        h3 = torch.clamp_min(h2, 0)
        h4 = torch.clamp_max(h3, 6)
        h5 = h4 / 6
        return h5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2)
