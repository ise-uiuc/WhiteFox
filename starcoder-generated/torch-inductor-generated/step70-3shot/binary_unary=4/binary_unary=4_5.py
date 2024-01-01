
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 2)
 
    def forward(self, x1, other=None):
        x2 = self.linear(x1)
        if other is not None:
                x2 += other
        v1 = x2.relu()
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(4, 5)
