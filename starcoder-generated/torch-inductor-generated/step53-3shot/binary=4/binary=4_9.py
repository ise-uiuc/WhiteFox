
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 4)
 
    def forward(self, x2, other=None):
        if other is None:
            other = torch.tensor([2.1, 3.1, 4.1, 5.1])
        v1 = self.linear(x2)
        v2 = v1 + other
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 2)
