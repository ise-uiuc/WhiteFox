
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 128)
 
    def forward(self, x1, x2):
        t1 = self.linear(x1)
        t2 = t1 + other
        t3 = t2 - t2 * t2
        return t3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 8)
x2 = torch.randn(2, 8)
