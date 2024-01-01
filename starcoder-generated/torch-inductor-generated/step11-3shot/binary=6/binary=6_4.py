
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(256)
 
    def forward(self, x1):
        y = self.linear(x1)
        t = x1 * (y.cos() - y.sin())
        return t

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 256)
