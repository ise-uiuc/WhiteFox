
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        self.linear.weight.data = torch.tensor([[1000]])
        self.linear.bias.data = torch.tensor([[-1000]])
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - other
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(4, 1)
