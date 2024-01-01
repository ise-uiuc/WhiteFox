
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(1584, 43, bias=False)
        self.other = torch.tensor(other, requires_grad=False)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + self.other
        return v2

# Initializing the model
m = Model(torch.rand(43))

# Inputs to the model
x1 = torch.randn(5, 43)
