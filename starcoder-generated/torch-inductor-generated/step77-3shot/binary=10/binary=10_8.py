
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 8)
 
        self.v1 = torch.randn(8, 8)
 
    def forward(self, x1, other=None):
        t1 = self.linear(x1)
        if other is not None:
            t1 += other
        return t1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16)
