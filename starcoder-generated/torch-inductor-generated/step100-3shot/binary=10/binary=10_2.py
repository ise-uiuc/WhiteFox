
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x1, other):
        return self.linear(x1) + other

# Initializing the model
other = torch.Tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
