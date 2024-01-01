
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(7, 8, bias=False)
 
    def forward(self, x1, other):
        v0 = self.linear(x1)
        v1 = v0 - other
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 7)
other = torch.Tensor(1)
other.requires_grad = True
