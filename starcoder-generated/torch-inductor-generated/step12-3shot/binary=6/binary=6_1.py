
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(160, 50)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - torch.nn.Parameter(torch.empty(1, 50).uniform_(-0.1, 0.1))
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 160)
