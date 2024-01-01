
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 4)
 
    def forward(self, x1, x2=None):
        if x2 is None:
            x2 = torch.empty(2, 3)
        v1 = self.linear(x1)
        v2 = v1 + x2
        return v2

# Initializing the model parameters
m = Model()

# Inputs to the model
x1 = torch.rand(3, 3)
x2 = torch.rand(2, 3, requires_grad=True)
