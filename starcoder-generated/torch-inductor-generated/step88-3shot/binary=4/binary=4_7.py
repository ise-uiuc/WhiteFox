
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 32)
 
    def forward(self, x1, x2):
        # t1 = linear(x1)
        t1 = self.linear(x1)
        # t2 = t1 + x2
        t2 = t1 + x2
        return t2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(4, 16)
x2 = torch.randn(4, 32)
