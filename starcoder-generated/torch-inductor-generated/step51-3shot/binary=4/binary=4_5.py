
class Model(torch.nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        self.other = torch.nn.Parameter(torch.load(b), requires_grad=False)
        self.bias = torch.nn.Parameter(torch.load(b), requires_grad=False)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + self.other
        v3 = v2 + self.bias
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.arange(1, 2).reshape([1, 1])
