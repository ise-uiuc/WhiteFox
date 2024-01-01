
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 1)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + other
        return v2


# Initializing the model
# Hint: Please set the name of other argument correctly when initializing the model.
other = torch.nn.Parameter(torch.randn(7, 16).uniform_(-0.1, 0.1, dtype=torch.float32), requires_grad=True)
m = Model()

# Inputs to the model
x1 = torch.randn(1, 7, 16)
